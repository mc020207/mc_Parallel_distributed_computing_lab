#include <bits/stdc++.h>
#include <sys/time.h>
#include <unistd.h>
#include "mpi.h"
#include <chrono>
using namespace std;
int N=1000000;
void phase1(int *array,int *localArray,int localSize,int *pivots,int numThr){
    // 每一个处理器进行局部的排序
    MPI_Scatter(array,localSize,MPI_INT,localArray,localSize,MPI_INT,0,MPI_COMM_WORLD);
    sort(localArray,localArray+localSize);
    for (int i=0;i<numThr;i++){
        pivots[i]=localArray[(i*(N/(numThr*numThr)))];
    }
}
void phase2(int *localArray,int localSize,int *pivots,int *partitionSizes,int numThr,int myId){
    int *collectedPivots=(int*)malloc(numThr*numThr*sizeof(pivots[0]));
    int *phase2Pivots=(int*)malloc((numThr-1)*sizeof(pivots[0]));
    int index=0;
    // 收集所有的采样
    MPI_Gather(pivots,numThr,MPI_INT,collectedPivots,numThr,MPI_INT,0,MPI_COMM_WORLD);
    if (myId == 0){
        // 选取主元
        sort(collectedPivots,collectedPivots+numThr*numThr);
        for (int i=0;i<(numThr-1);i++){
            phase2Pivots[i]=collectedPivots[(((i+1)*numThr)+(numThr/2))-1];
        }
    }
    MPI_Bcast(phase2Pivots,numThr-1,MPI_INT,0,MPI_COMM_WORLD);
    // 根据选取的主元，每一个处理器将自己范围的数分成numThr份给出每一份含有的元素数量。
    for (int i=0;i<localSize;i++){
        while (index<numThr-1&&localArray[i]>phase2Pivots[index]) index+=1;
        if (index==numThr-1){
            partitionSizes[numThr-1]=localSize-i;
            break;
        }
        partitionSizes[index]++;
    }
    free(collectedPivots);
    free(phase2Pivots);
    return;
}

void phase3(int *localArray,int *partitionSizes,int **newLocalArray,int *newPartitionSizes,int numThr){
    int totalSize=0;
    int *sendDisp=(int*)malloc(numThr*sizeof(int));
    int *recvDisp=(int*)malloc(numThr*sizeof(int));
    // 进行全局的通信，由此可以在一个处理器中获得每一个处理器分给他处理的元素个数
    MPI_Alltoall(partitionSizes,1,MPI_INT,newPartitionSizes,1,MPI_INT,MPI_COMM_WORLD);
    for (int i=0;i<numThr;i++) totalSize+=newPartitionSizes[i];
    *newLocalArray=(int*)malloc(totalSize*sizeof(int));
    // 计算发送位置和接受位置的数组用于传递真实的元素值
    sendDisp[0]=0;
    recvDisp[0]=0;
    for (int i=1;i<numThr;i++){
       sendDisp[i]=partitionSizes[i-1]+sendDisp[i-1];
       recvDisp[i]=newPartitionSizes[i-1]+recvDisp[i-1];
    }
    MPI_Alltoallv(&(localArray[0]),partitionSizes,sendDisp,MPI_INT,*newLocalArray,newPartitionSizes,recvDisp,MPI_INT,MPI_COMM_WORLD);
    free(sendDisp);
    free(recvDisp);
    return;
}

void phase4(int *newLocalArray,int *partitionSizes,int numThr,int myId,int *array){
    int *sortedSubList;
    int *recvDisp,*indexes,*partitionEnds,*subListSizes,totalListSize;
    indexes=(int*)malloc(numThr*sizeof(int));
    partitionEnds=(int*)malloc(numThr*sizeof(int));
    indexes[0]=0;
    totalListSize=partitionSizes[0];
    for (int i=1;i<numThr;i++){
        totalListSize+=partitionSizes[i];
        indexes[i]=indexes[i-1]+partitionSizes[i-1];
        partitionEnds[i-1]=indexes[i];
    }
    partitionEnds[numThr-1]=totalListSize;
    sortedSubList=(int*)malloc(totalListSize*sizeof(int));
    subListSizes=(int*)malloc(numThr*sizeof(int));
    recvDisp=(int*)malloc(numThr*sizeof(int));
    // 归并排序
    for (int i=0;i<totalListSize;i++){
        int lowest=INT_MAX;
        int ind=-1;
        for (int j=0;j<numThr;j++){
            if ((indexes[j]<partitionEnds[j]) && (newLocalArray[indexes[j]]<lowest)){
                lowest=newLocalArray[indexes[j]];
                ind=j;
            }
        }
        sortedSubList[i]=lowest;
        indexes[ind]+=1;
    }
    MPI_Gather(&totalListSize,1,MPI_INT,subListSizes,1,MPI_INT,0,MPI_COMM_WORLD);
    if (myId == 0){
        recvDisp[0]=0;
        for (int i=1;i<numThr;i++){
            recvDisp[i]=subListSizes[i-1]+recvDisp[i-1];
        }
    }
    MPI_Gatherv(sortedSubList,totalListSize,MPI_INT,array,subListSizes,recvDisp,MPI_INT,0,MPI_COMM_WORLD);
    free(partitionEnds);
    free(sortedSubList);
    free(indexes);
    free(subListSizes);
    free(recvDisp);
    return;
}
int main(int argc,char *argv[]){
    srand(0);
    MPI_Init(&argc,&argv);
    int numThr,myId,*partitionSizes,*newPartitionSizes,nameLength;
    int localSize,startIndex,endIndex,*pivots,*newLocalArray;
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto start = currentTime.time_since_epoch();
    auto end = currentTime.time_since_epoch();
    MPI_Comm_size(MPI_COMM_WORLD,&numThr);
    MPI_Comm_rank(MPI_COMM_WORLD,&myId);
    int *array=NULL;
    int *realarray=NULL;
    int addnum=0;
    if (myId==0){
        array=(int*)malloc((N+numThr)*sizeof(int));
        realarray=(int*)malloc((N+numThr)*sizeof(int));
        int i;
        for (i=0;i<N;i++) array[i]=realarray[i]=rand();
        currentTime = std::chrono::high_resolution_clock::now();
        start = currentTime.time_since_epoch();
        sort(realarray,realarray+N);
        currentTime = std::chrono::high_resolution_clock::now();
        end = currentTime.time_since_epoch();
        cout<<"Serial computation:"<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1e9<<"s\n";
        for (;i%numThr;i++) array[i]=INT_MIN;
        addnum=i-N;
        N=i;
    }
    for (;N%numThr;N++);
    MPI_Barrier(MPI_COMM_WORLD);
    currentTime = std::chrono::high_resolution_clock::now();
    start = currentTime.time_since_epoch();
    pivots=(int*)malloc(numThr*sizeof(int));
    partitionSizes=(int*)malloc(numThr*sizeof(int));
    newPartitionSizes=(int*)malloc(numThr*sizeof(int));
    for (int k=0;k<numThr;k++) partitionSizes[k]=0;
    localSize=(N+numThr-1)/numThr;
    int *localArray=(int*)malloc(localSize*sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);
    phase1(array,localArray,localSize,pivots,numThr);
    if (numThr>1){
        phase2(localArray,localSize,pivots,partitionSizes,numThr,myId);
        phase3(localArray,partitionSizes,&newLocalArray,newPartitionSizes,numThr);
        phase4(newLocalArray,newPartitionSizes,numThr,myId,array);
    }
    if (myId == 0){
        for(int k=addnum;k<N;k++) array[k-addnum]=array[k];
        currentTime = std::chrono::high_resolution_clock::now();
        end = currentTime.time_since_epoch();
        cout<<"Parallel computing:"<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1e9<<"s\n";
        cout<<"check\n";
        for (int i=0;i<N-addnum;i++){
            assert(realarray[i]==array[i]);
        }
    }
    if (numThr>1) free(newLocalArray);
    free(partitionSizes);
    free(newPartitionSizes);
    free(pivots);
    free(array);
    MPI_Finalize();
    return 0;
}