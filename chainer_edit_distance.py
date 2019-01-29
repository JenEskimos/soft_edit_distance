import numpy as np
from chainer import Variable, cuda
import chainer.functions as F
import cupy
from chainer import Function

_forward_kernel = cuda.elementwise(
    'raw T x1, raw T x2, raw Z l1, raw z l2, Z max_l, Z n_symbol, T temp',
    'raw T alpha, raw T beta',
    """
        int offset = i * (max_l + 1) * (max_l + 1);
        float exp_1t = exp(-temp);
        float exp_2t = exp(-2 * temp);
        beta[offset] = 1;
        alpha[offset] = 0;
        for(int j = 1; j < l1[i] + 1; j++){
            beta[offset + j * (max_l + 1)] = beta[offset + (j - 1) * (max_l + 1)] * exp_1t;
            alpha[offset + j * (max_l + 1)] = j * beta[offset + j * (max_l + 1)];
        }

        for(int k = 1; k < l2[i] + 1; k++){
            beta[offset + k] = beta[offset + k - 1] * exp_1t;
            alpha[offset + k] = k * beta[offset + k];
        }

        for(int j = 1; j < l1[i] + 1; j++){
            for(int k = 1; k < l2[i] + 1; k++){
                int index = offset + j * (max_l + 1) + k;
                T C = 0;
                int offset1 = i * max_l * n_symbol + j - 1;
                int offset2 = i * max_l * n_symbol + k - 1;
                for(int r = 0; r < n_symbol; r++)
                    C += abs(x1[offset1 + r * max_l] - x2[offset2 + r * max_l]) * 0.5;
                T expC = exp(-C * temp);
                T S = beta[index - (max_l + 1)] + beta[index - 1];

                alpha[index] = (alpha[index - max_l - 2] + beta[index - max_l - 2] * C) * expC +
                                (S + alpha[index - max_l - 1] + alpha[index - 1]) * exp_1t - 
                                (alpha[index - max_l - 2] + beta[index - max_l - 2] * 2) * exp_2t;
                beta[index] = beta[index - max_l - 2] * (expC - exp_2t) + exp_1t * S;
            }  
        } 
    """,
    name='forward_edit_dist_kernel'
)

_diff_x2_kernel = cuda.elementwise(
    'raw T x1, raw T x2, raw Z l1, raw z l2, Z max_l, Z n_symbol',
    'L dCdX',
    """
        int i0 = i / (max_l * max_l * n_symbol);
        int i1 = (i % (max_l * max_l * n_symbol)) / (max_l * max_l);
        int i2 = (i % (max_l * max_l)) / max_l;
        int i3 = i % max_l;

        if(i2 < l1[i0])
            if(i3 < l2[i0])
                if(x1[i0 * max_l * n_symbol + i1 * max_l + i2] > x2[i0 * max_l * n_symbol + i1 * max_l + i3])
                    dCdX = - 0.5;
                else
                    dCdX = 0.5;
    """,
    name='diff_x2_kernel'
)

_backward_kernel = cuda.elementwise(
    'raw T x1, raw T x2, raw L dCdX, raw T alpha, raw T beta, raw Z l1, raw Z l2, Z max_l, Z n_symbol, T temp',
    'raw T dX, raw T dAlpha, raw T dBeta',
    """

        float gap = exp(- temp);
        float exp_2t = exp(- 2 * temp);

        int i0 = i / (max_l * n_symbol);
        int m = (i % (max_l * n_symbol)) / max_l;
        int k = i % max_l;

        int offset_d = i0 * (max_l + 1) * max_l * n_symbol * 2 + m * max_l + k;
        int offset = i0 * (max_l + 1) * (max_l + 1);
        int offset_dCdX = i0 * max_l * max_l * n_symbol + m * max_l * max_l + k;

        if(k < l2[i0]){
            for(int i1 = 1; i1 < l1[i0] + 1; i1++){
                T C = 0;
                int offset1 = i0 * max_l * n_symbol + i1 - 1;
                int offset2 = i0 * max_l * n_symbol + k;
                for(int r = 0; r < n_symbol; r++)
                    C += abs(x1[offset1 + r * max_l] - x2[offset2 + r * max_l]) * 0.5;
                T expC = exp(-C * temp);

                T b_up = dBeta[offset_d + (i1 - 1) * (2 * max_l * n_symbol)];
                T a_up = dAlpha[offset_d + (i1 - 1) * (2 * max_l * n_symbol)];

                dBeta[offset_d + i1 * (2 * max_l * n_symbol)] = b_up * gap
                 - temp * beta[offset + (i1 - 1) * (max_l + 1) + k] * dCdX[offset_dCdX + (i1 - 1) * max_l] * expC;
                dAlpha[offset_d + i1 * (2 * max_l * n_symbol)] = (b_up + a_up) * gap
                 + dCdX[offset_dCdX + (i1 - 1) * max_l] * expC * (
                     - temp * (beta[offset + (i1 - 1) * (max_l + 1) + k] * C +
                      alpha[offset + (i1 - 1) * (max_l + 1) + k]) + beta[offset + (i1 - 1) * (max_l + 1) + k]);

            }

            for(int j = k; j < l2[i0] - 1; j++){
                for(int i1 = 1; i1 < l1[i0] + 1; i1++){
                    T C = 0;
                    int offset1 = i0 * max_l * n_symbol + i1 - 1;
                    int offset2 = i0 * max_l * n_symbol + j + 1;
                    for(int r = 0; r < n_symbol; r++)
                        C += abs(x1[offset1 + r * max_l] - x2[offset2 + r * max_l]) * 0.5;
                    T expC = exp(-C * temp);

                    int index0 = offset_d + i1 * (2 * max_l * n_symbol) + n_symbol * max_l;
                    T b_up = dBeta[offset_d + (i1 - 1) * (2 * max_l * n_symbol) + n_symbol * max_l];
                    T b_left = dBeta[offset_d + i1 * (2 * max_l * n_symbol)];
                    T b_d = dBeta[offset_d + (i1 - 1) * (2 * max_l * n_symbol)];

                    T a_up = dAlpha[offset_d + (i1 - 1) * (2 * max_l * n_symbol) + n_symbol * max_l];

                    dBeta[index0] = b_d * (expC - exp_2t) + (b_left + b_up) * gap;        
                    dAlpha[index0] = (b_d * C + dAlpha[offset_d + (i1 - 1) * (2 * max_l * n_symbol)]) * expC + 
                        (b_left + b_up + dAlpha[offset_d + i1 * (2 * max_l * n_symbol)] + a_up) * gap - 
                        (dAlpha[offset_d + (i1 - 1) * (2 * max_l * n_symbol)] + 2 * dBeta[offset_d + (i1 - 1)
                         * (2 * max_l * n_symbol)]) * exp_2t;

                    dBeta[offset_d + (i1 - 1) * (2 * max_l * n_symbol)] = b_up;
                    dAlpha[offset_d + (i1 - 1) * (2 * max_l * n_symbol)] = a_up;
                }

                dBeta[offset_d + l1[i0] * (2 * max_l * n_symbol)] =
                    dBeta[offset_d + l1[i0] * (2 * max_l * n_symbol) + n_symbol * max_l];
                dAlpha[offset_d + l1[i0] * (2 * max_l * n_symbol)] =
                    dAlpha[offset_d + l1[i0] * (2 * max_l * n_symbol) + n_symbol * max_l];
            }
            if(k < l2[i0] - 1){
                T last_dB = dBeta[offset_d + l1[i0] * (2 * max_l * n_symbol) + n_symbol * max_l];
                T last_dA = dAlpha[offset_d + l1[i0] * (2 * max_l * n_symbol) + n_symbol * max_l];
                T last_B = beta[offset + l1[i0] * (max_l + 1) + l2[i0]];
                T last_A = alpha[offset + l1[i0] * (max_l + 1) + l2[i0]];
                dX[i] = (last_dA * last_B - last_A * last_dB) / (last_B * last_B);
            }
            else{
                T last_dB = dBeta[offset_d + l1[i0] * (2 * max_l * n_symbol)];
                T last_dA = dAlpha[offset_d + l1[i0] * (2 * max_l * n_symbol)];
                T last_B = beta[offset + l1[i0] * (max_l + 1) + l2[i0]];
                T last_A = alpha[offset + l1[i0] * (max_l + 1) + l2[i0]];
                dX[i] = (last_dA * last_B - last_A * last_dB) / (last_B * last_B);
            }
        }
    """,
    name='backward_edit_dist_kernel'
)


class EditDistance(Function):
    def __init__(self, temp=1):
        super(EditDistance, self).__init__()
        self.temp = temp

    def forward_cpu(self, inputs):
        pass

    def backward_cpu(self, inputs, grad_outputs):
        pass

    def forward_gpu(self, inputs):
        x1, x2 = inputs
        l1 = cupy.sum((cupy.sum(x1, axis=1) > 0).astype(np.int32), axis=1)
        l2 = cupy.sum((cupy.sum(x2, axis=1) > 0).astype(np.int32), axis=1)
        alpha = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        beta = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        _forward_kernel(x1, x2, l1, l2, x1.shape[2], x1.shape[1], self.temp, alpha, beta, size=len(x1))
        alpha = alpha[list(range(len(l1))), l1, l2]
        beta = beta[list(range(len(l1))), l1, l2]
        d = alpha / beta

        alpha = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        beta = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        _forward_kernel(x1, x1, l1, l1, x1.shape[2], x1.shape[1], self.temp, alpha, beta, size=len(x1))
        alpha = alpha[list(range(len(l1))), l1, l1]
        beta = beta[list(range(len(l1))), l1, l1]
        d11 = alpha / beta

        alpha = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        beta = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        _forward_kernel(x2, x2, l2, l2, x1.shape[2], x1.shape[1], self.temp, alpha, beta, size=len(x1))
        alpha = alpha[list(range(len(l1))), l2, l2]
        beta = beta[list(range(len(l1))), l2, l2]
        d22 = alpha / beta

        return d - 0.5 * (d11 + d22),

    def backward_gpu(self, inputs, grad_outputs):
        x1, x2 = inputs
        dX = cupy.zeros(x2.shape, dtype=np.float32)
        g, = grad_outputs

        l1 = cupy.sum((cupy.sum(x1, axis=1) > 0).astype(np.int32), axis=1)
        l2 = cupy.sum((cupy.sum(x2, axis=1) > 0).astype(np.int32), axis=1)
        alpha = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        beta = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        _forward_kernel(x1, x2, l1, l2, x1.shape[2], x1.shape[1], self.temp, alpha, beta, size=len(x1))

        dCdX = cupy.zeros((x1.shape[0], x1.shape[1], x1.shape[2], x2.shape[2]), dtype=np.float32)
        _diff_x2_kernel(x1, x2, l1, l2, x1.shape[2], x1.shape[1], dCdX)

        dA = cupy.zeros((x1.shape[0], x1.shape[2] + 1, 2, x1.shape[1], x1.shape[2]), dtype=np.float32)
        dB = cupy.zeros((x1.shape[0], x1.shape[2] + 1, 2, x1.shape[1], x1.shape[2]), dtype=np.float32)

        _backward_kernel(x1, x2, dCdX, alpha, beta, l1, l2, x1.shape[2], x1.shape[1], self.temp, dX, dA, dB,
                         size=dX.size)

        dCdX = cupy.zeros((x1.shape[0], x1.shape[1], x2.shape[2], x2.shape[2]), dtype=np.float32)
        _diff_x2_kernel(x2, x2, l2, l2, x1.shape[2], x1.shape[1], dCdX)
        dCdX[:, :, np.arange(x2.shape[2]), np.arange(x2.shape[2])] = 0
        alpha = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        beta = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        _forward_kernel(x2, x2, l2, l2, x1.shape[2], x1.shape[1], self.temp, alpha, beta, size=len(x2))

        dA = cupy.zeros((x1.shape[0], x1.shape[2] + 1, 2, x1.shape[1], x1.shape[2]), dtype=np.float32)
        dB = cupy.zeros((x1.shape[0], x1.shape[2] + 1, 2, x1.shape[1], x1.shape[2]), dtype=np.float32)

        dX2 = cupy.zeros(x2.shape, dtype=np.float32)
        _backward_kernel(x2, x2, dCdX, alpha, beta, l2, l2, x1.shape[2], x1.shape[1], self.temp, dX2, dA, dB,
                         size=dX2.size)
        dX = dX - 0.5 * dX2

        dX = dX.T * g
        return None, dX.T


def soft_edit_distance(x1, x2, temp=1):
    return EditDistance(temp)(x1, x2)


def edit_distance(x1, x2, l1=None, l2=None, normolized=False):
    if len(x1.shape) == 3:
        kernel = cupy.ElementwiseKernel(
            'raw T x1, raw T x2, raw Z l1, raw Z l2, Z max_l, Z n_symbol',
            'raw T d',
            """
            int offset = i * (max_l + 1) * (max_l + 1);
            for(int j = 0; j < l1[i] + 1; j++)
                d[offset + j * (max_l + 1)] = j;

            for(int k = 0; k < l2[i] + 1; k++)
                d[offset + k] = k;

            for(int j = 1; j < l1[i] + 1; j++){
                for(int k = 1; k < l2[i] + 1; k++){
                    int index = offset + j * (max_l + 1) + k;
                    T delta = 0;
                    int offset1 = i * max_l * n_symbol + j - 1;
                    int offset2 = i * max_l * n_symbol + k - 1;
                    for(int r = 0; r < n_symbol; r++)
                        delta += max(x1[offset1 + r * max_l] - x2[offset2 + r * max_l], 0.0);

                    d[index] = min(d[index - (max_l + 1)] + 1, min(d[index - 1] + 1, d[index - max_l - 2] + delta));
                }  
            } 
            """,
            name='edit_distance_kernel_W'
        )
        if l1 is None:
            l1 = cupy.sum((cupy.sum(x1, axis=1) > 0).astype(np.int32), axis=1)
            l2 = cupy.sum((cupy.sum(x2, axis=1) > 0).astype(np.int32), axis=1)
        d = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
        kernel(x1, x2, l1, l2, x1.shape[2], x1.shape[1], d, size=len(x1))

    else:
        kernel = cupy.ElementwiseKernel(
            'raw T x1, raw T x2, raw Z l1, raw Z l2, Z max_l',
            'raw T d',
            """
            int offset = i * (max_l + 1) * (max_l + 1);
            for(int j = 0; j < l1[i] + 1; j++)
                d[offset + j * (max_l + 1)] = j;

            for(int k = 0; k < l2[i] + 1; k++)
                d[offset + k] = k;

            for(int j = 1; j < l1[i] + 1; j++){
                for(int k = 1; k < l2[i] + 1; k++){
                    int index = offset + j * (max_l + 1) + k;
                    T delta = 0;
                    if(x1[i * max_l + j - 1] != x2[i * max_l + k - 1])
                        delta = 1;
                    d[index] = min(d[index - (max_l + 1)] + 1, min(d[index - 1] + 1, d[index - max_l - 2] + delta));
                }  
            } 
            """,
            name='edit_distance_kernel'
        )
        if l1 is None:
            l1 = cupy.sum((x1 > 0).astype(np.int32), axis=1)
            l2 = cupy.sum((x2 > 0).astype(np.int32), axis=1)
        if x1.shape[1] < 255:
            dtype = np.uint8
        else:
            if x1.shape[1] < 65535:
                dtype = np.uint16
            else:
                dtype = np.uint32
        d = cupy.zeros((x1.shape[0], x1.shape[1] + 1, x1.shape[1] + 1), dtype=dtype)
        kernel(x1.astype(d.dtype), x2.astype(d.dtype), l1.astype(d.dtype), l2.astype(d.dtype), x1.shape[1], d,
               size=len(x1))

    d = d[list(range(len(l1))), l1, l2]
    if normolized:
        d = d - cupy.abs(l1 - l2)
    return d


def inrepolate(x1, x2):
    kernel = cupy.ElementwiseKernel(
        'raw T x1, raw T x2, raw T e, raw Z l1, raw Z l2, Z max_l, Z n_symbol',
        'raw T d, raw T m, raw T x3',
        """
        int offset = i * (max_l + 1) * (max_l + 1);
        for(int j = 0; j < l1[i] + 1; j++)
            d[offset + j * (max_l + 1)] = j;

        for(int k = 0; k < l2[i] + 1; k++)
            d[offset + k] = k;

        for(int j = 1; j < l1[i] + 1; j++){
            for(int k = 1; k < l2[i] + 1; k++){
                int index = offset + j * (max_l + 1) + k;
                T delta = 0;
                int offset1 = i * max_l * n_symbol + j - 1;
                int offset2 = i * max_l * n_symbol + k - 1;
                for(int r = 0; r < n_symbol; r++)
                    delta += max(x1[offset1 + r * max_l] - x2[offset2 + r * max_l], 0.0);
                m[index] = delta;
                d[index] = min(d[index - (max_l + 1)] + 1, min(d[index - 1] + 1, d[index - max_l - 2] + delta));
            }  
        }
        int j = l1[i];
        int k = l2[i];
        int t = max_l - 1;
        while(j > 0 && k > 0){
            int offset1 = i * max_l * n_symbol + j - 1;
            int offset2 = i * max_l * n_symbol + k - 1;
            int offset3 = i * max_l * n_symbol + t;
            int index = offset + j * (max_l + 1) + k;
            float score = d[index]; 
            if(score == d[index - max_l - 2] + m[index]){
                for(int r = 0; r < n_symbol; r++)
                    x3[offset3 + r * max_l] = (1 - e[index]) * x1[offset1 + r * max_l] + e[index] * x2[offset2 + r * max_l];
                k --;
                j --;
                t --;
            }
            else{
                if(score == d[index - (max_l + 1)] + 1){
                    if(e[index] < 0.5){
                        for(int r = 0; r < n_symbol; r++)
                            x3[offset3 + r * max_l] = x1[offset1 + r * max_l];
                        t --;
                    }
                    j --;
                }
                else{
                    if(e[index] >= 0.5){
                        for(int r = 0; r < n_symbol; r++)
                            x3[offset3 + r * max_l] = x2[offset2 + r * max_l];
                        t --;
                    }
                    k --;
                }
            }
        }
        while(j > 0){
            int offset1 = i * max_l * n_symbol + j - 1;
            int offset3 = i * max_l * n_symbol + t;
            int index = offset + j * (max_l + 1) + k;
            if(e[index] < 0.5){
                for(int r = 0; r < n_symbol; r++)
                    x3[offset3 + r * max_l] = x1[offset1 + r * max_l];
                t --;
            }
            j --;
        }
        while(k > 0){
            int offset2 = i * max_l * n_symbol + k - 1;
            int offset3 = i * max_l * n_symbol + t;
            int index = offset + j  * (max_l + 1) + k;
            if(e[index] >= 0.5){
                for(int r = 0; r < n_symbol; r++)
                    x3[offset3 + r * max_l] = x2[offset2 + r * max_l];
                t --;
            }
            k --;
        }
        """,
        name='interpolate_kernel'
    )
    shift_kernel = cupy.ElementwiseKernel(
        'raw T x, raw Z l, Z max_l, Z n_symbol',
        'raw T y',
        """
            int i0 = i / (max_l * n_symbol);
            int i1 = (i % (max_l * n_symbol)) / max_l;
            int i2 = i % max_l;
            if(i2 + max_l - l[i0] < max_l)
                y[i] = x[i0 * max_l * n_symbol + i1 * max_l + i2 + max_l - l[i0]];
        """,
        name='shift_kernel'
    )
    l1 = cupy.sum((cupy.sum(x1, axis=1) > 0).astype(np.int32), axis=1)
    l2 = cupy.sum((cupy.sum(x2, axis=1) > 0).astype(np.int32), axis=1)

    d = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
    m = cupy.zeros((x1.shape[0], x1.shape[2] + 1, x1.shape[2] + 1), dtype=np.float32)
    x3 = cupy.zeros(x1.shape, dtype=np.float32)
    e = cupy.random.uniform(0, 1, d.shape, dtype=np.float32)
    kernel(x1, x2, e, l1, l2, x1.shape[2], x1.shape[1], d, m, x3, size=len(x1))
    l3 = cupy.sum((cupy.sum(x3, axis=1) > 0).astype(np.int32), axis=1)
    out = cupy.zeros(x1.shape, dtype=np.float32)
    shift_kernel(x3, l3, x1.shape[2], x1.shape[1], out, size=x3.size)
    return out
