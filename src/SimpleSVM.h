#ifndef _SIMPLESVM_H
#define _SIMPLESVM_H

#include <iostream>
#include <thread>
#include <vector>
#include "Timer.h"
#include <assert.h>
#include <algorithm>

void train_singlethread(const double ** data, double * labels, double * model, int nex, int nfeat, double lr, double cpos, double cneg){

  for(int k=0;k<nex;k++){
    //int i = indices[k];
    int i = k;
    const double * example = data[i];
    double label = labels[i];

    double dot = model[nfeat-1];

    for(int j=0;j<nfeat-1;j++){
      dot += example[j] * model[j];
    }

    double sampleweight = (label == 1) ? cpos : cneg;

    if( 1 - label * dot > 0){
      for(int j=0;j<nfeat;j++){
        model[j] = (1.0 - 1.0 / nex * lr) * model[j] - lr * sampleweight * (-label * example[j]) * (1 - label * dot);
      }
      model[nfeat-1] = (1.0 - 1.0 / nex * lr) * model[nfeat-1] - lr * sampleweight * (-label) * (1 - label * dot);
    }
  }

}

double loss_singlethread(const double ** data, double * labels, double * model, int nex, int nfeat, double cpos, double cneg){
  double loss = 0.0;

  double modle_norm = 0.0;
  for(int i=0;i<nfeat;i++){
    modle_norm += model[i]*model[i];
  }

  for(int i=0;i<nex;i++){
    const double * example = data[i];
    double label = labels[i];
    double dot = model[nfeat-1];
    for(int j=0;j<nfeat-1;j++){
      dot += example[j] * model[j];
    }
    if (isnan(dot)){
      return 100000000000000000.0;
    }
    
    double sampleweight = (label == 1) ? cpos : cneg;

    if(1 - label * dot > 0){
      loss +=  sampleweight * (1 - label * dot) * (1 - label * dot) + 0.5 * 1.0 / nex * modle_norm;
    }
    
  }
  return loss / nex;
}

void trainSVM(const double ** data, double * labels, int nex, int nfeat, int nworker, double * weights, int * scan_order, double cpos, double cneg){
  std::cerr << "start training " << "with " << nworker << " workers..."  << cpos << " " << cneg << std::endl;
  Timer timer2;

  /*
  std::cout << "# " << nex << std::endl;
  std::cout << "# " << nfeat << std::endl;
  */

  std::vector<double *> models;
  for(int i=0;i<nworker;i++){
    double * model = new double[nfeat + 16];  // pad by 16 to avoid false sharing
    models.push_back(model);
    for(int j=0;j<nfeat;j++){
      models[i][j] = 0.0;
    }
  }

  double original_loss = loss_singlethread(data, labels, models[0], nex, nfeat, cpos, cneg);
  
  double bestloss = -1;
  double lrs[4] = {0.1, 0.001, 0.0001, 0.00001};
  int ilr;
  for(ilr = 0; ilr < 4; ilr ++){
    for(int i=0;i<nworker;i++){
      for(int j=0;j<nfeat;j++){
        models[i][j] = 0.0;
      }
    }
    double lr = lrs[ilr];
    int NEPOCH = 10;
    for(int iepoch=0;iepoch<NEPOCH;iepoch++){
      lr = lrs[ilr] / (iepoch + 1);
      int data_per_partition = nex / nworker;
      std::vector<std::thread> threads;
      for(int i=0;i<nworker;i++){
        if( i != nworker - 1){
          threads.push_back(std::thread(train_singlethread, &data[i * data_per_partition], 
              &labels[i * data_per_partition], models[0], data_per_partition, nfeat, lr, cpos, cneg));
        }else{
          threads.push_back(std::thread(train_singlethread, &data[i * data_per_partition], 
              &labels[i * data_per_partition], models[0], nex - (nworker - 1) * data_per_partition, nfeat, lr, cpos, cneg));
        }
      }
      for(int i=0;i<nworker;i++){
        threads[i].join();
      }
      /*
      for(int j=0;j<nfeat;j++){
        for(int i=1;i<nworker;i++){
          models[0][j] += models[i][j];
        }
      }
      */
      /*
      for(int j=0;j<nfeat;j++){
        models[0][j] = models[0][j] / nworker;
        for(int i=1;i<nworker;i++){
          models[i][j] = models[0][j];
        }
      }
      */
    }

    double real_loss = loss_singlethread(data, labels, models[0], nex, nfeat, cpos, cneg);
    if(original_loss > real_loss && (real_loss < bestloss || bestloss < 0)){
      bestloss = real_loss;
      for(int i=0;i<nfeat;i++){
        weights[i] = models[0][i];
      }
      break;
    }
      
  }

  assert(bestloss != -1);

  std::cerr << "final loss = " << bestloss << " lr = " << lrs[ilr] << std::endl;

  for(int i=0;i<nworker;i++){
    delete[] models[i];
  }

  std::cerr << "finish training: " << timer2.elapsed() << " seconds. " << nex << " examples." << std::endl;

}


#endif
