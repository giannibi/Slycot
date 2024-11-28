import slycot as sly
import numpy as np
from numpy.testing import assert_array_less

# For computing the inverse of scalings
def invss(A,B,C,D):
    # inverse of D
    dinv = np.linalg.inv(D)
    # inverse system matrices
    ainv = A - B @ dinv @ C
    binv = B @ dinv
    cinv = -dinv @ C

    return ainv, binv, cinv, dinv

def cascss(A1,B1,C1,D1,A2,B2,C2,D2):
     n, _ = np.shape(A1)
     _, m = np.shape(A2)
     A = np.block([[A1,np.zeros(n,m)],[B2*C1,A2]])
     B = np.block([[B1],[B2*D1]])
     C = np.block([D2*C1,C2])
     D = D2*D1
     return A,B,C,D


def myhinfsyn(A,B,C,D, nmeas, ncon, initgamma=1e6):
    """H_{inf} control synthesis for plant P.

    Parameters
    ----------
    P: partitioned lti plant
    nmeas: number of measurements (input to controller)
    ncon: number of control inputs (output from controller)
    initgamma: initial gamma for optimization

    Returns
    -------
    K: controller to stabilize P (State-space sys)
    CL: closed loop system (State-space sys)
    gam: infinity norm of closed loop system
    rcond: 4-vector, reciprocal condition estimates of:
        1: control transformation matrix
        2: measurement transformation matrix
        3: X-Riccati equation
        4: Y-Riccati equation
    """

    n = np.size(A, 0)
    m = np.size(B, 1)
    np_ = np.size(C, 0)
    out = sly.sb10ad(n, m, np_, ncon, nmeas, initgamma, A, B, C, D)
    gam = out[0]
    Ak = out[1]
    Bk = out[2]
    Ck = out[3]
    Dk = out[4]
    Ac = out[5]
    Bc = out[6]
    Cc = out[7]
    Dc = out[8]
    rcond = out[9]

    return Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, gam, rcond


def musyn(AG, BG, CG, DG, f, nblock, itype, omega, maxiter=10, qutol=2, order=4, initgamma=1e6, verbose=False):
      '''
      Perform mu synthesis using D-K iteration
      
      K, best_nubar, init_mubar, best_mubar, gamma 
               = musyn(G, f, nblock, itype, omega, maxiter=10, qutol=2, order=4, verbose=True)
               
      *G:       LFT form of the system from [wdelta,u] to [zdelta,y]
      f:       controller input-output dimension
      nblock:  uncertainty structure (vector of sizes of uncertain blocks)
      itype:   must be 2 for each entry of nblock, other values not implemented
      omega:   frequency vector for D scaling computation
      maxiter: max number of iterations
      qutol:   tolerance for sb10md routine, play with it only if you get numerical problems
      order:   order of the scalings D(j*omega), increase it to try to get more accurate results (unlikely)
      verbose: print iteration info
      K:       controller
      best_nubar:
               best achieved upper bound to mu norm of Tzw_delta (best achieved nubar) 
      init_mubar:
               mu upper bound at the first iteration (as function of frequency)
      best_mubar:
               achieved mu upper bound at the last iteration (as function of frequency)
      gamma:   closed loop norm achieved by initial Hinf controller
      '''
      # Initial K-step: compute an initial optimal Hinf controller without D scaling
      Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, gamma, rcond = myhinfsyn(G, f, f, initgamma)
      if verbose:
            print("Infinity norm of Tzw_delta with initial Hinfinity controller: ", gamma)

      # Start with a best mu norm upper bound
      # slightly higher than the achieved gamma without scaling
      best_nubar = gamma * 1.001
      i = 1
      while(True):
            if verbose:
                print("Iteration #", i)
            # D-step: compute optimal scalings for the current closed loop
            # and the corresponding upper bound mubar vs. frequency
            # If numerical problems occur, try reducing the order of the closed loop cl0
            # for the sake of computing the scaling D. This may impact the performance of the final controller
            _, _, _, _, _, _, D_A, D_B, D_C, D_D, mubar, _ = sly.sb10md(f, order, nblock, itype, qutol, Ac, Bc, Cc, Dc, omega)
            if i == 1:
                  # Save the mubar of the first iteration
                  initial_mubar = mubar
                  best_mubar = mubar
            # Get current value of the peak of mubar, i.e., the current upper bound
            # to the mu norm
            sup_mubar = np.max(mubar)
            if sup_mubar >= best_nubar:
                  # The current iteration did not improve nubar over the previous ones
                  if verbose:
                      print("No better upper bound to mu norm of Tzw_delta found")
                  if i > 1:
                        mubar = best_mubar
            else:
                  # Save best upper bound so far
                  best_nubar = sup_mubar
                  if verbose:
                      print("Best upper bound to mu norm of Tzw_delta: ", best_nubar)
                  # And the best mubar
                  best_mubar = mubar
                  if i > 1:
                        # And the best controller
                        k = kb
            i = i+1
            if i > maxiter:
                  break 

            ADinv, BDinv, CDinv, DDinv = invss(D_A, D_B, D_C, D_D)
            # Compute D*G*(inv(D))
            A1,B1,C1,D1 = cascss(ADinv,BDinv,CDinv,DDinv,AG,BG,CG,DG)
            ADGDInv,BDGDInv,CDGDInv,DDGDInv = cascss(A1,B1,C1,D1,D_A,D_B,D_C,D_D)

            # K-step: compute controller for current scaling
            try: 
                  kb, cl0, gamma, rcond = myhinfsyn(ADGDInv,BDGDInv,CDGDInv,DDGDInv, f, f, initgamma)
            except:
                  # Something went wrong: keep last controller
                  kb = k

      return k, best_nubar, initial_mubar, best_mubar, gamma




class Test_sb10md():
      
     def test_sb10md_0(self):
          
     # LFT representation for the robust performance problem
      
      G_A = np.array([[-1.00000000e-06,  0.00000000e+00, -6.66666667e-02,
        -6.93889390e-18,  6.93889390e-18, -1.38777878e-17],
       [ 0.00000000e+00, -1.00000000e-06, -1.24571246e-17,
        -6.66666667e-02,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00, -1.33333333e-02,
        -3.56691510e-17, -4.45118870e-16,  2.18487185e-16],
       [ 0.00000000e+00,  0.00000000e+00, -1.50973436e-17,
        -1.33333333e-02,  2.45649572e-16,  3.01300498e-17],
       [ 0.00000000e+00,  0.00000000e+00,  1.29953970e-16,
        -1.72182401e-16, -2.00000000e+00,  2.74651719e-16],
       [ 0.00000000e+00,  0.00000000e+00, -1.07552856e-16,
         3.38271078e-17,  0.00000000e+00, -2.00000000e+00]])
      
      G_B = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -5.00000000e-02,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  5.00000000e-02,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 1.08200000e+00, -1.09600000e+00,  0.00000000e+00,
         0.00000000e+00,  1.08200000e+00, -1.09600000e+00],
       [-8.78000000e-01,  8.64000000e-01,  0.00000000e+00,
         0.00000000e+00, -8.78000000e-01,  8.64000000e-01],
       [-2.63083827e-17, -1.27004847e-16,  0.00000000e+00,
         0.00000000e+00, -2.63083827e-17,  1.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  0.00000000e+00]])
      
      G_C = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00, -3.60000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00, -3.60000000e+00, -8.88178420e-16],
       [ 0.00000000e+00,  9.99990000e-01, -2.19881109e-16,
        -6.66666667e-01,  4.16333634e-17,  0.00000000e+00],
       [-9.99990000e-01,  0.00000000e+00,  6.66666667e-01,
         0.00000000e+00,  0.00000000e+00, -5.55111512e-17],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.33333333e+00, -8.32667268e-17,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00, -1.33333333e+00,
         0.00000000e+00,  0.00000000e+00,  1.11022302e-16]])
      
      G_D = np.array([[ 0. ,  0. ,  0. ,  0. ,  2. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  2. ],
       [ 0. ,  0. ,  0.5,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0.5,  0. ,  0. ],
       [ 0. ,  0. , -1. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. , -1. ,  0. ,  0. ]])
      


      # Controller I/O sizes
      f = 2

      # Extended uncertainty structure: two 1x1 uncertainty blocks and a 2x2 performance block
      nblock = np.array([1,1,2])

      # This has to be == 2 (complex uncertainty) for each block (other values are not implemented)
      itype = np.array([2,2,2])

      # Frequency range for mu computations
      omega = np.logspace(-3, 3, 61)

      # Do mu-synthesis via D-K iteration
      K, best_nubar, init_mubar, best_mubar, gamma = musyn(G_A, G_B, G_C, G_D, f, nblock, itype, omega, order=4, qutol=1, initgamma=10)

      # Testing Assertion
      assert_array_less(best_nubar, 1.03)

