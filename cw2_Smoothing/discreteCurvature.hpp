//
//  discreteCurvature.hpp
//  
//
//  Created by Cinomi on 2018/3/11.
//

#ifndef discreteCurvature_hpp
#define discreteCurvature_hpp

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <math.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <igl/fit_plane.h>
#include <igl/viewer/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/triangle_triangle_adjacency.h>

#include "nanoflann.hpp"
#include "tutorial_shared_path.h"
#include "nanogui/formhelper.h"
#include "nanogui/screen.h"
#include "igl/jet.h"
#include "SymEigsSolver.h"
#include "GenEigsSolver.h"
#include "MatOp/SparseGenMatProd.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>

using namespace Eigen;
using namespace std;
using namespace nanoflann;
using namespace Spectra;


namespace discreteCurvature {
   SparseMatrix<double> uniform_laplacian(MatrixXd V, MatrixXi F);
   SparseMatrix<double> nonUniform_laplacian(MatrixXd V, MatrixXi F);
   MatrixXd mean_curvature(MatrixXd V, MatrixXi F, int discretization);
   MatrixXd gaussian_curvature(MatrixXd V, MatrixXi F);
   MatrixXd eigen_reconstruction(MatrixXd V, MatrixXi F, int k, int m);
   MatrixXd getNormals(MatrixXd V);
   double get_angle(RowVector3d P1, RowVector3d P2, RowVector3d P3);
   pair<double,double> get_area_angle(MatrixXd V, MatrixXi F, int idx);
   MatrixXcd get_eigenvecs(SparseMatrix<double> lapla, int k, int m);
   MatrixXd complex2real(MatrixXcd complex_mat);
}




// ========================================================== //
// ================== 1) Uniform Laplacian ================== //
// ========================================================== //
SparseMatrix<double> discreteCurvature::uniform_laplacian(MatrixXd V, MatrixXi F){
    // save memory for results
    Eigen::SparseMatrix<double> Lapla(V.rows(),V.rows());
    
    std::vector<int> uniq_neighbour;
    bool findV;
    int position, nei_N;
    float test;
    
    for(int i=0; i<V.rows(); i++){
        for(int j=0; j<F.rows(); j++){
            findV=false;
            
            // go through all verteces compose the face and find the current one
            for(int k=0; k<3; k++){
                if(F(j,k)==i){
                    position=k;
                    findV=true;
                    break;
                }
            }
       
            // save neighbours
            if(findV){
                uniq_neighbour.push_back(F(j,(position+1)%3));
                uniq_neighbour.push_back(F(j,(position+2)%3));
            }
        }
        
        // get size of neighbours
        nei_N = uniq_neighbour.size();
        
        // fill sparse matrix with neighbours
        for(int idx=0;idx<nei_N;idx++){
            Lapla.insert(i,uniq_neighbour[idx]) = 1.0/double(nei_N);
        }
        
        Lapla.insert(i,i)=-1.0;
        uniq_neighbour.clear();
    }

    return Lapla;
}



// ========================================================== //
// ================ 2) Non-uniform Laplacian ================ //
// ========================================================== //
SparseMatrix<double> discreteCurvature::nonUniform_laplacian(MatrixXd V, MatrixXi F){
    SparseMatrix<double> Lapla(V.rows(),V.rows()), C_mat(V.rows(),V.rows()), M_inv(V.rows(),V.rows()), M(V.rows(),V.rows());
    RowVector3d P1, P2, P3;
    int adjTri_idx, adjEdge_idx;
    double alpha_ij, beta_ij, weight_ij;
    
    MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);
    // TT: stores the index of neighboring triangle for triangle i
    // TTi: stores the index of edges of triangle in TT contact with triangle i
    
    for(int i=0; i<V.rows(); i++){
        pair<double,double> areaAngle = get_area_angle(V,F,i);
        double area = areaAngle.first / 3;
        double wij_sum = 0;
        
        int nextP, preP, edgeIdx;
        for(int j=0; j<F.rows(); j++){
            nextP = -1;
            preP = -1;
            edgeIdx = -1;
            if(F(j,0)==i){
                nextP = F(j,1);
                preP = F(j,2);
                edgeIdx = 2;
            }else if(F(j,1)==i){
                nextP = F(j,2);
                preP = F(j,0);
                edgeIdx = 0;
            }else if(F(j,2)==i){
                nextP = F(j,0);
                preP = F(j,1);
                edgeIdx = 1;
            }
            
            // if any neighbor is found
            if(edgeIdx!=-1){
                P1 = V.row(i);
                P2 = V.row(nextP);
                P3 = V.row(preP);
                
                // compute angel between vertecies
                beta_ij = get_angle(P1,P2,P3);
                
                // find adjacne triangles
                adjTri_idx = TT(j,edgeIdx);
                adjEdge_idx = TTi(j,edgeIdx);
                
                RowVector3i adjF = F.row(adjTri_idx);
                if(adjEdge_idx==0){
                    P1 = V.row(adjF(0));
                    P2 = V.row(adjF(1));
                    P3 = V.row(adjF(2));
                }else if(adjEdge_idx==1){
                    P1 = V.row(adjF(1));
                    P2 = V.row(adjF(2));
                    P3 = V.row(adjF(0));
                }else if(adjEdge_idx==2){
                    P1 = V.row(adjF(2));
                    P2 = V.row(adjF(0));
                    P3 = V.row(adjF(1));
                }
                
                // compute angle: alpha_ij
                alpha_ij = get_angle(P1,P3,P2);
                
                // compute cotan weight w_ij
                //weight_ij = tan(M_PI/2-alpha_ij)+tan(M_PI/2-beta_ij);
                weight_ij = cos(alpha_ij)/sin(alpha_ij) + cos(beta_ij)/sin(beta_ij);
                
                // fill laplacian matrix
                C_mat.insert(i,preP)=weight_ij;
                wij_sum = wij_sum+weight_ij;
            }
            
        }
        C_mat.insert(i,i)=-wij_sum;
        M.insert(i,i)=2.0*area;
        M_inv.insert(i,i)=1.0/(2.0*area);
        
    }
    
    // compute laplacian operator
    Lapla = M_inv * C_mat;
    
    return Lapla;
}



// ========================================================== //
// ===================== Mean Curvature ===================== //
// ========================================================== //
Eigen::MatrixXd discreteCurvature::mean_curvature(MatrixXd V, MatrixXi F, int discretization){
    // ----- get normals
    Eigen::MatrixXd N(V.rows(),3);
    //MatrixXd N;
    //igl::per_face_normals(V,F,N);
    N = getNormals(V);
  
    // ----- compute required Laplacian Operator
    Eigen::SparseMatrix<double> Lapla(V.rows(),V.rows());
    if(discretization==0){
        Lapla = uniform_laplacian(V,F);
    }else if(discretization==1){
        Lapla = nonUniform_laplacian(V,F);
    }

    // ----- compute mean curvature
    Eigen::VectorXd H = 0.5*(Lapla*V).rowwise().norm();
    
    // ----- orient mean curvature
    std::vector<int> uniq_neighbour;
    bool findV;
    int position, nei_N;
    for(int i=0; i<V.rows(); i++){
        for(int j=0; j<F.rows(); j++){
            findV=false;
            
            // go through all verteces compose the face and find the current one
            for(int k=0; k<3; k++){
                if(F(j,k)==i){
                    position=k;
                    findV=true;
                    break;
                }
            }
            
            // save neighbours
            if(findV){
                uniq_neighbour.push_back(F(j,(position+1)%3));
                uniq_neighbour.push_back(F(j,(position+2)%3));
            }
        }
        
        // get size of neighbours
        nei_N = uniq_neighbour.size();
        RowVector3d average;
        average.setZero();
        
        // fill sparse matrix with neighbours
        for(int idx=0;idx<nei_N;idx++){
            RowVector3d currentRow = V.row(uniq_neighbour[idx]);
            average = average + currentRow/nei_N;
        }
        
        if(N.row(i).dot(average-V.row(i))>0){
            H.row(i)=-H.row(i);
        }
        uniq_neighbour.clear();
    }
    
    // set mean curvature as color matrix
    Eigen::MatrixXd C(F.rows(),3);
    igl::jet(H,true,C);
    
    return C;
}


// ========================================================== //
// =================== Gaussian Curvature =================== //
// ========================================================== //
MatrixXd discreteCurvature::gaussian_curvature(MatrixXd V, MatrixXi F){
    Eigen::VectorXd H(V.rows());
    
    for(int i=0; i<V.rows(); i++){
        pair<double,double> areaAngle = get_area_angle(V, F, i);
        double area = areaAngle.first;
        H(i) = areaAngle.second / (area/3.0);
    }
    Eigen::MatrixXd C(F.rows(),3);
    igl::jet(H,true,C);
    
    return C;
}



// ========================================================== //
// ===================== Reconstruction ===================== //
// ========================================================== //
MatrixXd discreteCurvature::eigen_reconstruction(MatrixXd V, MatrixXi F, int k, int m){
    // ----- compute Laplacian operator
    SparseMatrix<double> Lapla(V.rows(), V.rows());
    Lapla = nonUniform_laplacian(V, F);
    
    // ----- compute k smallest eigenvectors
    MatrixXcd complex_eigenvecs = get_eigenvecs(Lapla, k, m);
    MatrixXd real_eigenvecs = complex2real(complex_eigenvecs);
    
    // ----- reconstruction
    MatrixXd recons_V(V.rows(),3);
    recons_V.setZero();
    
    MatrixXd scaler(1,3);
    for(int i=0; i<real_eigenvecs.cols(); i++){
        scaler(0,0) = (V.col(0).transpose()) * real_eigenvecs.col(i);
        scaler(0,1) = (V.col(1).transpose()) * real_eigenvecs.col(i);
        scaler(0,2) = (V.col(2).transpose()) * real_eigenvecs.col(i);
        
        recons_V.col(0) = recons_V.col(0) + scaler(0,0) * real_eigenvecs.col(i);
        recons_V.col(1) = recons_V.col(1) + scaler(0,1) * real_eigenvecs.col(i);
        recons_V.col(2) = recons_V.col(2) + scaler(0,2) * real_eigenvecs.col(i);
    }

    return recons_V;
}
/*
MatrixXd discreteCurvature::eigen_reconstruction(MatrixXd V, MatrixXi F, int k, int m){
    // ----- compute Laplacian operator
    SparseMatrix<double> Lapla(V.rows(), V.rows());
    Lapla = nonUniform_laplacian(V, F);

    // ----- compute k smallest eigenvectors
    MatrixXcd complex_eigenvecs = get_eigenvecs(Lapla, k, m);
    MatrixXd real_eigenvecs = complex2real(complex_eigenvecs);

    // ----- reconstruction

    MatrixXd recons_V(V.rows(), 3);
    recons_V.setZero();
    for(int i=0; i<real_eigenvecs.cols(); i++){
        recons_V.col(0) += (V.col(0).transpose() * real_eigenvecs.col(i)) * real_eigenvecs.col(i);
        recons_V.col(1) += (V.col(1).transpose() * real_eigenvecs.col(i)) * real_eigenvecs.col(i);
        recons_V.col(2) += (V.col(2).transpose() * real_eigenvecs.col(i)) * real_eigenvecs.col(i);
    }

    std::cout<<"K= "<<k<<endl;
    std::cout<<"diffX= "<<(V.col(0)-recons_V.col(0)).array().abs().colwise().sum()<<endl;
    std::cout<<"diffY= "<<(V.col(1)-recons_V.col(1)).array().abs().colwise().sum()<<endl;
    std::cout<<"diffZ= "<<(V.col(2)-recons_V.col(2)).array().abs().colwise().sum()<<endl;
    
    return recons_V;
}
*/



// ========================================================== //
// ======= Angle between given vertecies Computation ======== //
// ========================================================== //
double discreteCurvature::get_angle(RowVector3d P1, RowVector3d P2, RowVector3d P3){
    return acos((P1-P2).dot(P3-P2)/((P1-P2).norm()*(P3-P2).norm()));
}



// ========================================================== //
// ============ Area & Angle deficit Computation ============ //
// ========================================================== //
pair<double,double> discreteCurvature::get_area_angle(MatrixXd V, MatrixXi F, int idx){
    RowVector3d P1, P2, P3;
    double angle_deflicit = 2.0*M_PI;
    double area_sum = 0;
    
    //----- get faces with current vertex
    for(int j=0; j<F.rows(); j++){
        RowVector3i currentF = F.row(j);
        bool neighFound = false;
        if (currentF(0)==idx){
            P1 = V.row(currentF(0));
            P2 = V.row(currentF(1));
            P3 = V.row(currentF(2));
            neighFound=true;
        }else if(currentF(1)==idx){
            P1 = V.row(currentF(1));
            P2 = V.row(currentF(2));
            P3 = V.row(currentF(0));
            neighFound=true;
        }else if(currentF(2)==idx){
            P1 = V.row(currentF(2));
            P2 = V.row(currentF(0));
            P3 = V.row(currentF(1));
            neighFound=true;
        }
        
        //----- compute area sum and angle deficit
        if(neighFound){
            double theta = get_angle(P2,P1,P3);
            double area = 0.5*(P2-P1).norm()*(P3-P1).norm()*sin(theta);
            area_sum = area_sum+area;
            angle_deflicit = angle_deflicit-theta;
        }
    }
    return make_pair(area_sum,angle_deflicit);
}



// ========================================================== //
// =================== Normals Estimation =================== //
// ========================================================== //
MatrixXd discreteCurvature::getNormals(MatrixXd V){
    int pts_num = V.rows();
    Eigen::MatrixXd normal(pts_num,3);
    Eigen::MatrixXd meancenter(1,3);
    meancenter = V.colwise().sum() / double(V.rows());
    
    KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(V, 10);
    kd_tree_index.index->buildIndex();
    
    for(int idx=0; idx<pts_num; idx++){
        std::vector<double> query_pt(3);
        
        for(size_t d=0; d<3; d++){
            query_pt[d]=V(idx,d);
        }
        
        // closest 8 neighbors
        const size_t results_num=8;
        vector<size_t> ret_index(results_num);
        vector<double> out_dists_sqr(results_num);
        
        // result set
        nanoflann::KNNResultSet<double> resultSet(results_num);
        
        resultSet.init(&ret_index[0], &out_dists_sqr[0] );
        kd_tree_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
        
        // get cloest 8 neighbor points index
        Eigen::MatrixXd selectPts(results_num,3);
        for(size_t i=0; i<results_num; i++){
            selectPts(i,0) = V(ret_index[i],0);
            selectPts(i,1) = V(ret_index[i],1);
            selectPts(i,2) = V(ret_index[i],2);
        }
        
        Eigen::RowVector3d Nvt, Ct;
        igl::fit_plane(selectPts, Nvt, Ct);
        
        normal(idx,0)=Nvt(0);
        normal(idx,1)=Nvt(1);
        normal(idx,2)=Nvt(2);
        
        //------ check the direction of normal vector
        if((meancenter(0,0)-V(idx,0)) * normal(idx,0) + (meancenter(0,1)-V(idx,1)) * normal(idx,1) + (meancenter(0,2)-V(idx,2)) * normal(idx,2) > 0) {
            normal(idx,0) = -Nvt(0);
            normal(idx,1) = -Nvt(1);
            normal(idx,2) = -Nvt(2);
        }
    }
    return normal;
}



// ========================================================== //
// ==================== Get Eigenvectors ==================== //
// ========================================================== //
MatrixXcd discreteCurvature::get_eigenvecs(SparseMatrix<double> lapla, int k, int m){
    SparseGenMatProd<double> op(lapla);
    GenEigsSolver< double, SMALLEST_MAGN, SparseGenMatProd<double> > eigs(&op, k, m);
    
    eigs.init();
    int nconv = eigs.compute();
    
    VectorXcd evals;
    MatrixXcd evecs;
    
    if(eigs.info() == SUCCESSFUL){
        evals = eigs.eigenvalues();
        evecs = eigs.eigenvectors();
        std::cout<<"Eigenvalues found:\n"<<evals<<endl;
    }
    
    return evecs;
}



// ========================================================== //
// ================ convert complex to real ================= //
// ========================================================== //
MatrixXd discreteCurvature::complex2real(MatrixXcd complex_mat){
    MatrixXd real_mat(complex_mat.rows(), complex_mat.cols());
    for(int i=0; i<complex_mat.rows(); i++){
        for(int j=0; j<complex_mat.cols(); j++){
            real_mat(i,j)=complex_mat(i,j).real();
        }
    }
    return real_mat;
}





#endif /* discreteCurvature_hpp */
