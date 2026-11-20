// Chapter 30.6 -- An optimistic weight-dominated dense-layer model.
// This is a prefill-shaped MHA layer, not a complete decode cost model.
// The FFN is SwiGLU: gate, up and down projections (three matrices).
// Norms, nonlinearities, softmax, KV/activation traffic, launch overhead
// and cache effects are excluded. Aggregate intensity is not a proof
// that every serial unfused kernel has the same bottleneck.
// Compile: g++ -std=c++23 -Wall -Wextra -O2 06_full_transformer_layer_roofline_analysis.cpp -o 06_full_transformer_layer_roofline_analysis
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
static int g_tests=0,g_passed=0;
#define CHECK(...) do { ++g_tests; if (__VA_ARGS__) ++g_passed; else std::cerr<<"FAIL line "<<__LINE__<<"\n"; } while(0)
bool near(double a,double b,double eps=1e-9){return std::fabs(a-b)<eps;}
struct LayerShape {int64_t hidden,heads,head_dim,ffn_dim,seq_len;};
struct FlopBreakdown {double qkv=0,attn=0,out_proj=0,ffn=0,total=0;};
struct ByteBreakdown {double qkv=0,out_proj=0,ffn=0,total=0;};
FlopBreakdown layer_flops(const LayerShape&s,int64_t batch){
 const double m=double(batch)*s.seq_len,h=double(s.hidden),f=double(s.ffn_dim),n=double(s.seq_len);
 FlopBreakdown x;
 x.qkv=6*m*h*h; x.attn=4*batch*n*n*h; x.out_proj=2*m*h*h;
 x.ffn=6*m*h*f; // gate h->f, up h->f, down f->h
 x.total=x.qkv+x.attn+x.out_proj+x.ffn;return x;
}
ByteBreakdown layer_weight_bytes(const LayerShape&s){
 const double h=double(s.hidden),f=double(s.ffn_dim);
 ByteBreakdown x; x.qkv=3*h*h*4; x.out_proj=h*h*4;
 x.ffn=3*h*f*4; x.total=x.qkv+x.out_proj+x.ffn;return x;
}
double aggregate_weight_intensity(const LayerShape&s,int64_t batch){return layer_flops(s,batch).total/layer_weight_bytes(s).total;}
double aggregate_weight_crossover(const LayerShape&s,double ridge){return ridge/aggregate_weight_intensity(s,1);}
// Each modeled serial kernel has time >= max(F_i/P, B_i/BW).
// This is still only a lower bound; omitted traffic makes it optimistic.
struct KernelCost {double flops,bytes;};
double serial_kernel_time_bound(const std::vector<KernelCost>& costs,double peak_flops,double bandwidth){
 double total=0;for(auto c:costs)total+=std::max(c.flops/peak_flops,c.bytes/bandwidth);return total;
}
int main(){
 std::cout<<"Chapter 30.6: SwiGLU layer accounting and roofline bounds\n";
 LayerShape s{4,2,2,8,2};
 auto f=layer_flops(s,1);auto b=layer_weight_bytes(s);
 CHECK(near(f.qkv,192));CHECK(near(f.attn,64));CHECK(near(f.out_proj,64));CHECK(near(f.ffn,384));CHECK(near(f.total,704));
 CHECK(near(b.qkv,192));CHECK(near(b.out_proj,64));CHECK(near(b.ffn,384));CHECK(near(b.total,640));
 CHECK(near(aggregate_weight_intensity(s,1),1.1));
 std::cout<<"  tiny MHA prefill shape: hidden=4, heads=2, head_dim=2, FFN=8, sequence=2\n";
 std::cout<<"  FLOPs: QKV=192, attention=64, output=64, SwiGLU projections=384; total=704\n";
 std::cout<<"  weight bytes: QKV=192, output=64, SwiGLU=384; total=640\n";
 std::cout<<"  aggregate FLOPs/weight-byte=704/640=1.1 (omitted traffic is not zero)\n";
 CHECK(near(layer_flops(s,2).total,1408));CHECK(near(aggregate_weight_intensity(s,2),2.2));
 const double crossover=aggregate_weight_crossover(s,2.0);
 CHECK(near(crossover,20.0/11.0));
 std::cout<<"  batch=2: modeled FLOPs=1408, weight bytes=640, aggregate intensity=2.2\n";
 std::cout<<"  aggregate weight-only crossover at ridge 2: "<<std::setprecision(6)<<crossover<<"\n";
 // Counterexample: serial kernels can be compute-limited and bandwidth-
 // limited separately even when aggregate intensity equals the ridge.
 std::vector<KernelCost> costs={{100,1},{1,100}};
 const double p=10,bw=10;
 const double aggregate=std::max(101.0/p,101.0/bw);
 const double serial=serial_kernel_time_bound(costs,p,bw);
 CHECK(near(aggregate,10.1));CHECK(near(serial,20));CHECK(serial>aggregate);
 std::cout<<"  illustrative kernels (F,B)=(100,1),(1,100), P=BW=10 units/s:\n";
 std::cout<<"    aggregate lower bound=10.1 s; sum of serial kernel bounds=20 s\n";
 std::cout<<"  These are analytical examples, not measured execution times or a whole-layer bottleneck classifier.\n";
 std::cout<<g_passed<<"/"<<g_tests<<" checks passed\n"<<(g_passed==g_tests?"ALL CHECKS PASSED\n":"SOME CHECKS FAILED\n");
 return g_passed==g_tests?0:1;
}
