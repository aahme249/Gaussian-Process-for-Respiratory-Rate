from .cg import pcg
from .inference import BBMMSettings, bbmm_marginal_log_likelihood, bbmm_predict
from .mbcg import mbcg
from .operators import (ExactKernelOperator, LinearKernelOperator, RBFKernel,
                        SoRKernelOperator)
from .pivoted_cholesky import PivotedCholeskyPreconditioner, pivoted_cholesky
from .slq import slq_logdet
