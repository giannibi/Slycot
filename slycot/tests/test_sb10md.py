import slycot as sly
import numpy as np
import control as ct
from numpy.testing import assert_array_less

# For computing the inverse of scalings
def invss(d):
    # inverse of D
    dinv = np.linalg.inv(d.D)
    # inverse system matrices
    ainv = d.A - d.B @ dinv @ d.C
    binv = d.B @ dinv
    cinv = -dinv @ d.C

    return ct.StateSpace(ainv, binv, cinv, dinv)


def myhinfsyn(P, nmeas, ncon, initgamma=1e6):
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

    n = np.size(P.A, 0)
    m = np.size(P.B, 1)
    np_ = np.size(P.C, 0)
    out = sly.sb10ad(n, m, np_, ncon, nmeas, initgamma, P.A, P.B, P.C, P.D)
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

    K = ct.StateSpace(Ak, Bk, Ck, Dk)
    CL = ct.StateSpace(Ac, Bc, Cc, Dc)

    return K, CL, gam, rcond


def musyn(G, f, nblock, itype, omega, maxiter=10, qutol=2, order=4, reduce=0, initgamma=1e6, verbose=False):
      '''
      Perform mu synthesis using D-K iteration
      
      K, best_nubar, init_mubar, best_mubar, gamma 
               = musyn(G, f, nblock, itype, omega, maxiter=10, qutol=2, order=4, reduce=0, verbose=True)
               
      G:       LFT form of the system from [wdelta,u] to [zdelta,y]
      f:       controller input-output dimension
      nblock:  uncertainty structure (vector of sizes of uncertain blocks)
      itype:   must be 2 for each entry of nblock, other values not implemented
      omega:   frequency vector for D scaling computation
      maxiter: max number of iterations
      qutol:   tolerance for sb10md routine, play with it only if you get numerical problems
      order:   order of the scalings D(j*omega), increase it to try to get more accurate results (unlikely)
      reduce:  if > 0, do a model reduction on the closed loop at each iteration for the sake of computing 
               the scaling D; set it to something lower than the full order of cl0 if you run into numerical
               problems (may impact performance of the final controller)
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
      k, cl0, gamma, rcond = myhinfsyn(G, f, f, initgamma)
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
            if reduce > 0:
                cl0 = ct.balred(cl0, reduce, method='truncate')
            _, _, _, _, _, _, D_A, D_B, D_C, D_D, mubar, _ = sly.sb10md(f, order, nblock, itype, qutol, cl0.A, cl0.B, cl0.C, cl0.D, omega)
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

            D = ct.StateSpace(D_A, D_B, D_C, D_D)
            # Compute D*G*(inv(D))
            DGDInv =  ct.minreal(D * G * invss(D), verbose = False)

            # K-step: compute controller for current scaling
            try: 
                  kb, cl0, gamma, rcond = myhinfsyn(DGDInv, f, f, initgamma)
            except:
                  # Something went wrong: keep last controller
                  kb = k

      return k, best_nubar, initial_mubar, best_mubar, gamma




class Test_sb10md():
      
     def test_sb10md_0(self):
          
      # Plant transfer function P
      gain = ct.StateSpace([],[],[],np.array([[87.8, -86.4],
                                          [108.2, -109.6]]))
      dyn = ct.tf2ss(ct.tf([1],[75,1]))
      P = ct.append(dyn,dyn) * gain

      # Plant input and output labels
      P.input_labels = ['up[0]','up[1]']
      P.output_labels = ['yp[0]','yp[1]']

      # Undertainty weight
      Wi = ct.tf2ss(ct.tf([1,0.2],[0.5,1]))
      Wi = ct.append(Wi, Wi)
      Wi.output_labels = ['zdelta[0]', 'zdelta[1]']

      # Performance weight on sensitivity
      Wp = 0.5*ct.tf2ss(ct.tf([10,1],[10,1e-5])) # table 8.2
      Wp = ct.append(Wp, Wp)
      Wp.input_labels = ['y[0]', 'y[1]']
      Wp.output_labels = ['z[0]', 'z[1]']

      # Summing junction into P
      sdelta = ct.summing_junction(inputs=['u','wdelta'], output='up', dimension=2)

      # Feedback summing junction
      fbk = ct.summing_junction(inputs=['w','-yp'], output='y', dimension=2)

      # Generate LFT for mu synthesis
      G = ct.interconnect([P, Wi, Wp, sdelta, fbk],
                        inputs=['wdelta[0]','wdelta[1]','w[0]','w[1]','u[0]','u[1]'],
                        outputs=['zdelta[0]','zdelta[1]','z[0]','z[1]','y[0]','y[1]'])


      # Controller I/O sizes
      f = 2

      # Extended uncertainty structure: two 1x1 uncertainty blocks and a 2x2 performance block
      nblock = np.array([1,1,2])

      # This has to be == 2 (complex uncertainty) for each block (other values are not implemented)
      itype = np.array([2,2,2])

      # Frequency range for mu computations
      omega = np.logspace(-3, 3, 61)

      # Do mu-synthesis via D-K iteration
      K, best_nubar, init_mubar, best_mubar, gamma = musyn(G, f, nblock, itype, omega, order=4, qutol=1, initgamma=10)

      # Testing Assertion
      assert_array_less(best_nubar, 1.03)

