#ifndef PID_H
#define PID_H

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */
  double Kp;
  double Ki;
  double Kd;
  //Twiddle variables
  double params[3];
  bool twiddle;
  double d_params[3];
  unsigned int loop_counter=0;
  unsigned int state_counter =0;
  double twd_error;
  bool initial_settle= false;
  double best_error =0;
  bool measurement=false;
  bool bad_measurement = false;
  unsigned int loop_count_max=100;

  /*
  * OLD CTE Errors
  */
  double prev_cte;
  bool error_initial;

  //double p_tht=0.2;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd, bool twiddle);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
