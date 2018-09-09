#include "PID.h"
#include <iostream>
using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd, bool twiddle) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  this->params[0] = Kp;
  this->params[1] = Ki;
  this->params[2] = Kd;
  this->twiddle = twiddle;
  this->d_params[0] = Kp*0.1;
  this->d_params[1] = Ki*0.1;
  this->d_params[2] = Kd*0.1;
}

void PID::UpdateError(double cte) {
  /*if(this->prev_cte == 0xffff){
    this->prev_cte = cte;
  }*/
  if(error_initial == false){
    error_initial = true;
    prev_cte = cte;
  }

  this->p_error = cte;
  this->i_error += cte;
  this->d_error =  cte - this->prev_cte;
  this->prev_cte = cte;
  /*Twiddle Implementation*/
  loop_counter +=1;
  //0-100 are settling cycles and 100-200 are measurement cycles for error measurement
  if(loop_counter > loop_count_max/2)
    twd_error += cte*cte ;
  //cout<<"update error"<<this->p_error<<" "<<this->i_error<<this->d_error<<endl;
  //for the initial robot settlement
  if(loop_counter==300 && initial_settle == false){
    initial_settle = true;
    best_error = twd_error/300;
    loop_counter =0;
    twd_error = 0;
  }
  //twiddle is ON and initial execution has been done
  if(twiddle == true && initial_settle == true && loop_counter == loop_count_max){
    loop_counter =0;
    cout<<"params = "<<" "<<params[0]<<" "<<params[1]<<" "<<params[2]<<endl;
    cout<<"d_params = "<<" "<<d_params[0]<<" "<<d_params[1]<<" "<<d_params[2]<<endl;
    //TODO chnage this threshold-condition for twidling
    if(d_params[0]+d_params[1]+d_params[2]> 0.015){
      if(measurement == false && bad_measurement== false){
        params[state_counter] += d_params[state_counter];
        measurement = true;
      }
      else{
        measurement = false;
        twd_error = twd_error/(loop_count_max/2);//mean squared error
        if(twd_error<best_error && bad_measurement == false){
          best_error = twd_error;
          d_params[state_counter] *= 1.1;
          //Goto next variable
          state_counter += 1;
          if(state_counter == 3)
            state_counter =0;
        }
        else{
          if(bad_measurement == false){
            params[state_counter] -= 2*d_params[state_counter];
            bad_measurement = true;
          }
          else{
            bad_measurement = false;
            if(twd_error< best_error){
              best_error = twd_error;
              d_params[state_counter] *=1.1;
            }
            else{
              params[state_counter] += d_params[state_counter];
              d_params[state_counter] *= 0.9;
            }
            //Goot next variable
            state_counter += 1;
            if(state_counter == 3)
              state_counter =0;
          }
        }
      }
    }
    else{
      cout<<"Miracle:: Params at hand  "<<params<<endl;
    }
  } //Twiddle loop end

}

double PID::TotalError() {
  cout<<"total_Error  p i d "<<this->Kp*this->p_error<<" "<<this->Ki*this->i_error<<" "<<this->Kd*this->d_error<<endl;
  return (this->params[0]*this->p_error) + (this->params[1]*this->i_error) \
   + (this->params[2]*this->d_error);//p_error + i_error + d_error; //
}
