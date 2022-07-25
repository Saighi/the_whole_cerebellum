#include <iostream>
#include <armadillo>
#include <math.h> 
#include <chrono>
#include <thread>

using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono; // nanoseconds, system_clock, seconds
using namespace arma;
using namespace std;

class Pendulum {
  public:

    int nb_nodes;
    double length_edge;
    mat pos_nodes;
    vec pos_angular;
    vec speed_angular;
    double acc_limit;
    double speed_limit;
    double delta_t;

    Pendulum() {};
    Pendulum(int nb_nodes_arg, double delta_t_arg) {

        delta_t = delta_t_arg;
        nb_nodes=nb_nodes_arg; 
        //speed_nodes = mat(2,nb_nodes, fill::zeros);
        pos_nodes = mat(2,nb_nodes);
        pos_angular= vec(nb_nodes);
        speed_angular= vec(nb_nodes);
        length_edge = 1./(nb_nodes-1);

        vec column = vec(2);
        for (int x=0; x<nb_nodes; x++){
            pos_angular(x)= datum::pi*3/2;
            column = {cos(pos_angular(x)),sin(pos_angular(x))};
            pos_nodes.col(x) = column*(length_edge*x);
            speed_angular(x)=0;
        }
        acc_limit = 0.1 ;
        speed_limit = 0.5 ;
    }

    void rotate(vec acc_nodes){
        //int size_acc = size(acc_nodes)(0);
               if (acc_nodes(0) != 0){
            cout<<"Acceleration of the first node should be zero (wall fixed pendulum)"<<endl;
        }
       
        vec column = vec(2);
        double pos_node_x;
        double pos_node_y;
        for(int x=1; x<nb_nodes; x++){
            speed_angular(x) = min(speed_angular(x)+(2*datum::pi*min(acc_nodes(x),acc_limit)), speed_limit*datum::pi);
            pos_angular(x)+= fmod(speed_angular(x),(2*datum::pi))*delta_t; // the impact of the speed on  the rotation depend on the delta_t of the simulation
            pos_node_x = pos_nodes.col(x-1)(0) + (cos(pos_angular(x))*length_edge);
            pos_node_y = pos_nodes.col(x-1)(1) + (sin(pos_angular(x))*length_edge);
            column = {pos_node_x,pos_node_y};
            pos_nodes.col(x) = column;
        } 

    }
};

int main(int argc, char **argv) {

    double delta_t = 0.01;
    int nb_nodes = 3;
    Pendulum pd = Pendulum(nb_nodes,delta_t);
    int nb_time_points = 1;
    mat all_node_x_traj = mat(nb_nodes,nb_time_points);
    mat all_node_y_traj= mat(nb_nodes,nb_time_points);

    cout << "position nodes" <<endl;
    cout << pd.pos_nodes <<endl;

    vec acc_nodes ={0, 0.05,0.05};
    for (int t=0; t<nb_time_points; t++){
        pd.rotate(acc_nodes);
        cout << "position nodes" <<endl;
        cout << pd.pos_nodes <<endl; 
        all_node_x_traj.col(t)=pd.pos_nodes.row(0).t();
        all_node_y_traj.col(t)=pd.pos_nodes.row(1).t();
        //sleep_for(nanoseconds(10000000));
        acc_nodes ={0,0,0};
    }
    all_node_x_traj.save("all_node_x_traj",raw_ascii);
    all_node_y_traj.save("all_node_y_traj",raw_ascii);



}