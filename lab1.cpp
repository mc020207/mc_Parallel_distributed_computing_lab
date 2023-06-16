# include<iostream>
# include<stdlib.h>
# include<assert.h>
# include<omp.h>
# include<algorithm>
# include<time.h>
# include<chrono>
# define MAXN 1000000
# define numThr 8
# define INF 2000000000
using namespace std;
int a[MAXN];
int b[MAXN];
int c[MAXN];
int bj[numThr+10];
int id[numThr+10];
int mxn[numThr],mt=0;
bool cmp(int i,int j){
    int x=id[i]<bj[i+1]?a[id[i]]:INF;
    int y=id[j]<bj[j+1]?a[id[j]]:INF;
    return x<y;
}
int main(){
    for (int i=0;i<MAXN;i++) a[i]=c[i]=(rand()<<15)+rand();
    cout<<"start\n";
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto start = currentTime.time_since_epoch();
    sort(c,c+MAXN);
    currentTime = std::chrono::high_resolution_clock::now();
    auto end = currentTime.time_since_epoch();
    cout<<"Serial computation:"<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1e9<<"s\n";
    currentTime = std::chrono::high_resolution_clock::now();
    start = currentTime.time_since_epoch();
    for (int i=1;i<=numThr;i++){
        bj[i]=bj[i-1]+MAXN/numThr+((MAXN%numThr)>(i-1));
        id[i]=bj[i];
        if (bj[i]-bj[i-1]) mxn[mt++]=i-1;
    }
    #pragma omp parallel num_threads(numThr)
    {
        sort(a+bj[omp_get_thread_num()],a+bj[omp_get_thread_num()+1]);
    }
    currentTime = std::chrono::high_resolution_clock::now();
    auto start2 = currentTime.time_since_epoch();
    cout<<"merge\n";
    sort(mxn,mxn+mt,cmp);
    for (int k=0;k<MAXN;k++){
        b[k]=a[id[mxn[0]]];
        id[mxn[0]]++;
        int now=mxn[0];
        if (id[now]<bj[now+1]){
        	int j; 
            for (j=1;j<mt;j++){
                if (a[id[mxn[j]]]<a[id[now]]) mxn[j-1]=mxn[j];
                else break;
            }
            mxn[j-1]=now;
        }else{
            for (int j=1;j<mt;j++) mxn[j-1]=mxn[j];
            mt--;
        }
    }
    currentTime = std::chrono::high_resolution_clock::now();
    end = currentTime.time_since_epoch();
    cout<<"Parallel computing:"<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1e9<<"s\n";
    cout<<"merge time:"<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-start2).count()/1e9<<"s\n";
    cout<<"check\n";
    for (int i=0;i<MAXN;i++) assert(b[i]==c[i]);
    cout<<"OK!!\n";
    return 0;
}
