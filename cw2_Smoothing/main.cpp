#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <math.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <igl/fit_plane.h>
#include <igl/viewer/Viewer.h>
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/principal_curvature.h>

#include "tutorial_shared_path.h"
#include "nanogui/formhelper.h"
#include "nanogui/screen.h"
#include "igl/jet.h"
#include "discreteCurvature.hpp"
#include "Smoothing.hpp"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

Eigen::MatrixXd V1,V2,V3,tem_V1,tem_V2,tem_V3;
Eigen::MatrixXi F1,F2,F3,tem_F1,tem_F2,tem_F3;

int main(int argc, char** argv){
    double lambda = 0.1;
    double noise = 0.1;
    int iteration = 1;
    int k = 1;
    
    enum Mesh_type {Bumpy=0,Cow,Bunny} type_mesh = Bumpy;
    enum Discretization_type {Uniform=0, Non_uniform} discretization = Uniform;
    
    igl::viewer::Viewer viewer;
    {
        viewer.core.show_lines=false;
        viewer.core.show_overlay=false;
        viewer.core.invert_normals=false;
    }
    
    igl::readOFF(TUTORIAL_SHARED_PATH "/bumpy.off",V1,F1);
    igl::readOFF(TUTORIAL_SHARED_PATH "/cow.off",V2,F2);
    igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off",V3,F3);
    
    igl::readOFF(TUTORIAL_SHARED_PATH "/bumpy.off",tem_V1,tem_F1);
    igl::readOFF(TUTORIAL_SHARED_PATH "/cow.off",tem_V2,tem_F2);
    igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off",tem_V3,tem_F3);

    // set color
    Eigen::MatrixXd C(F1.rows(),3);
    C<<
    Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1);
    
    viewer.data.set_mesh(V1,F1);
    viewer.data.set_colors(C);
    
    viewer.callback_init = [&type_mesh, &lambda, &iteration, &discretization, &k, &noise](igl::viewer::Viewer& viewer){
        viewer.ngui->addWindow(Eigen::Vector2i(900,10), "Coursework 2");
        
        // ============================================================= //
        // ==================== Test Mesh Selection ==================== //
        // ============================================================= //
        viewer.ngui->addGroup("Initialization");
        viewer.ngui->addVariable<Mesh_type>("Which mesh do you want to test?",type_mesh)->setItems({"Bumpy ","Cow ","Bunny "});
        viewer.ngui->addButton("Show it",[&](){
            viewer.data.clear();
            igl::readOFF(TUTORIAL_SHARED_PATH "/bumpy.off",V1,F1);
            igl::readOFF(TUTORIAL_SHARED_PATH "/cow.off",V2,F2);
            igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off",V3,F3);
            
            if(type_mesh==0){
                Eigen::MatrixXd C(F1.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1);
                
                viewer.data.set_mesh(V1,F1);
                viewer.data.set_colors(C);
            }else if(type_mesh==1){
                Eigen::MatrixXd C(F2.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(F2.rows(),1);
                
                viewer.data.set_mesh(V2,F2);
                viewer.data.set_colors(C);
            }else{
                Eigen::MatrixXd C(F3.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(F3.rows(),1);
                
                viewer.data.set_mesh(V3,F3);
                viewer.data.set_colors(C);
            }
            
        });
        
        
        // ============================================================= //
        // ================== 1) Uniform Discretization ================ //
        // ============================================================= //
        viewer.ngui->addGroup("Uniform Discretization");
        viewer.ngui->addButton("Mean Curvature",[&](){
            viewer.data.clear();
            MatrixXd C;
            if(type_mesh==0){
                C = discreteCurvature::mean_curvature(V1,F1,0);
                viewer.data.set_mesh(V1,F1);
                viewer.data.set_colors(C);
            }else if(type_mesh==1){
                C = discreteCurvature::mean_curvature(V2,F2,0);
                viewer.data.set_mesh(V2,F2);
                viewer.data.set_colors(C);
            }else{
                C = discreteCurvature::mean_curvature(V3,F3,0);
                viewer.data.set_mesh(V3,F3);
                viewer.data.set_colors(C);
            }
            
        });
        viewer.ngui->addButton("Gaussian Curvature",[&](){
            viewer.data.clear();
            MatrixXd C;
            if(type_mesh==0){
                C = discreteCurvature::gaussian_curvature(V1,F1);
                viewer.data.set_mesh(V1,F1);
                viewer.data.set_colors(C);
            }else if(type_mesh==1){
                C = discreteCurvature::gaussian_curvature(V2,F2);
                viewer.data.set_mesh(V2,F2);
                viewer.data.set_colors(C);
            }else{
                C = discreteCurvature::gaussian_curvature(V3,F3);
                viewer.data.set_mesh(V3,F3);
                viewer.data.set_colors(C);
            }
        });
    
        
        // ============================================================= //
        // ================ 2) Non-uniform Discretization ============== //
        // ============================================================= //
        viewer.ngui->addGroup("Non-uniform Discretization");
        viewer.ngui->addButton("Mean Curvature",[&](){
            viewer.data.clear();
            MatrixXd C;
            if(type_mesh==0){
                C = discreteCurvature::mean_curvature(V1,F1,1);
                viewer.data.set_mesh(V1,F1);
                viewer.data.set_colors(C);
            }else if(type_mesh==1){
                C = discreteCurvature::mean_curvature(V2,F2,1);
                viewer.data.set_mesh(V2,F2);
                viewer.data.set_colors(C);
            }else{
                C = discreteCurvature::mean_curvature(V3,F3,1);
                viewer.data.set_mesh(V3,F3);
                viewer.data.set_colors(C);
            }
        });
        
        
        // ============================================================= //
        // ===================== 3) Reconstruction ===================== //
        // ============================================================= //
        viewer.ngui->addGroup("Reconstruction");
        viewer.ngui->addVariable<int>("k",[&](int val){
            k = val;
        },[&](){
            return k;
        });
        viewer.ngui->addButton("Reconstruction",[&](){
            viewer.data.clear();
            if(type_mesh==0){
                MatrixXd re_V = discreteCurvature::eigen_reconstruction(V1, F1, k, 300);
                MatrixXd C(re_V.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(re_V.rows(),1);
                
                viewer.data.set_mesh(re_V,F1);
                viewer.data.set_colors(C);
                 
                //pair<VectorXcd,MatrixXcd> test_eigen = test_test(V1,F1);
            }else if(type_mesh==1){
                
                MatrixXd re_V = discreteCurvature::eigen_reconstruction(V2, F2, k, 300);
                MatrixXd C(re_V.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(re_V.rows(),1);
                
                viewer.data.set_mesh(re_V,F2);
                viewer.data.set_colors(C);
                
                //pair<VectorXcd,MatrixXcd> test_eigen = test_test(V2,F2);
            }else if(type_mesh==2){
                MatrixXd re_V = discreteCurvature::eigen_reconstruction(V3, F3, k, 300);
                MatrixXd C(re_V.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(re_V.rows(),1);
                
                viewer.data.set_mesh(re_V,F3);
                viewer.data.set_colors(C);
            }
                 
        });
        
        
        
        
        
        
        // ============================================================= //
        // ============ 4) Explicit Laplacian mesh smoothing =========== //
        // ============================================================= //
         viewer.ngui->addGroup("Laplacian Mesh Smoothing");
         viewer.ngui->addVariable<double>("Lambda",[&](double val){
             lambda = val;
         },[&](){
             return lambda;
         });
         viewer.ngui->addVariable<int>("Iteration", [&](int val){
             iteration = val;
         },[&](){
             return iteration;
         });
         viewer.ngui->addVariable<Discretization_type>("Discretization",discretization)->setItems({"Uniform ","Non-uniform "});
         viewer.ngui->addButton("Explicit Smoothing",[&](){
         viewer.data.clear();
         MatrixXd C;
         if(type_mesh==0){
             pair<MatrixXd,MatrixXd> V_color = Smoothing::explicit_smooth(V1,F1,discretization,lambda);
             for (int count=1; count<iteration; count++){
                 V_color = Smoothing::explicit_smooth(V_color.first,F1,discretization,lambda);
             }
             
             float error = Smoothing::compute_error(tem_V1, V_color.first);
             std::cout<<"error: "<<error<<endl;
             
             viewer.data.set_mesh(V_color.first,F1);
             viewer.data.set_colors(V_color.second);
         }else if(type_mesh==1){
             pair<MatrixXd,MatrixXd> V_color = Smoothing::explicit_smooth(V2,F2,discretization,lambda);
             for (int count=1; count<iteration; count++){
                 V_color = Smoothing::explicit_smooth(V_color.first,F2,discretization,lambda);
             }
             
             float error = Smoothing::compute_error(tem_V2, V_color.first);
             std::cout<<"error: "<<error<<endl;
             
             viewer.data.set_mesh(V_color.first,F2);
             viewer.data.set_colors(V_color.second);
         }else{
             pair<MatrixXd,MatrixXd> V_color = Smoothing::explicit_smooth(V3,F3,discretization,lambda);
             for (int count=1; count<iteration; count++){
                 V_color = Smoothing::explicit_smooth(V_color.first,F3,discretization,lambda);
             }
             
             float error = Smoothing::compute_error(tem_V3, V_color.first);
             std::cout<<"error: "<<error<<endl;
             
             viewer.data.set_mesh(V_color.first,F3);
             viewer.data.set_colors(V_color.second);
         }
         });
        
        
        // ============================================================= //
        // ============ 5) Implicit Laplacian mesh smoothing =========== //
        // ============================================================= //
        viewer.ngui->addButton("Implicit Smoothing",[&](){
            viewer.data.clear();
            MatrixXd C;
            if(type_mesh==0){
                pair<MatrixXd,MatrixXd> V_color = Smoothing::implicit_smooth(V1,F1,lambda);
                for (int count=1; count<iteration; count++){
                    V_color = Smoothing::implicit_smooth(V_color.first,F1,lambda);
                }
                
                float error = Smoothing::compute_error(tem_V1, V_color.first);
                std::cout<<"error: "<<error<<endl;
                
                viewer.data.set_mesh(V_color.first,F1);
                viewer.data.set_colors(V_color.second);
            }else if(type_mesh==1){
                pair<MatrixXd,MatrixXd> V_color = Smoothing::implicit_smooth(V2,F2,lambda);;
                for (int count=1; count<iteration; count++){
                    V_color = Smoothing::implicit_smooth(V_color.first,F2,lambda);
                }
                
                float error = Smoothing::compute_error(tem_V2, V_color.first);
                std::cout<<"error: "<<error<<endl;
                
                viewer.data.set_mesh(V_color.first,F2);
                viewer.data.set_colors(V_color.second);
            }else{
                pair<MatrixXd,MatrixXd> V_color = Smoothing::implicit_smooth(V3,F3,lambda);;
                for (int count=1; count<iteration; count++){
                    V_color = Smoothing::implicit_smooth(V_color.first,F3,lambda);
                }
                
                float error = Smoothing::compute_error(tem_V3, V_color.first);
                std::cout<<"error: "<<error<<endl;
                
                viewer.data.set_mesh(V_color.first,F3);
                viewer.data.set_colors(V_color.second);
            }
        });
        
        
        // ============================================================= //
        // ======================= 6) Add Noise ======================== //
        // ============================================================= //
        viewer.ngui->addVariable<double>("Noise Level",[&](double val){
            noise = val;
        },[&](){
            return noise;
        });
        viewer.ngui->addButton("Add Noise",[&](){
            viewer.data.clear();

            if(type_mesh==0){
                V1 = Smoothing::add_noise(V1, noise);
                MatrixXd C(V1.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(V1.rows(),1);
                
                float error = Smoothing::compute_error(tem_V1, V1);
                std::cout<<"error: "<<error<<endl;
                
                viewer.data.set_mesh(V1,F1);
                viewer.data.set_colors(C);
            }else if(type_mesh==1){
                V2 = Smoothing::add_noise(V2, noise);
                MatrixXd C(V2.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(V2.rows(),1);
                
                float error = Smoothing::compute_error(tem_V2, V2);
                std::cout<<"error: "<<error<<endl;
                
                viewer.data.set_mesh(V2,F2);
                viewer.data.set_colors(C);
            }else if(type_mesh==2){
                V3 = Smoothing::add_noise(V3, noise);
                MatrixXd C(V3.rows(),3);
                C<<
                Eigen::RowVector3d(0.9,0.775,0.25).replicate(V3.rows(),1);
                
                float error = Smoothing::compute_error(tem_V3, V3);
                std::cout<<"error: "<<error<<endl;
                
                viewer.data.set_mesh(V3,F3);
                viewer.data.set_colors(C);
            }
        });
        
        
        viewer.screen->performLayout();
        return false;
    };
    
    

    
    viewer.launch();
    return 0;
    
}

