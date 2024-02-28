Source code for our DAC 2024 paper, "Enabling Multiple Tensor-wise Operator Fusion for Transformer Models on Spatial Accelerators"

## Setup
1. Follow the instruction in gurobi website to get the gurobi licence 
2. Install c++ boost library.
3. Clone the repository.
```
git clone git@github.com:lxu28973/DAT.git
```
4. Compile the code.
```
cd DAT && cmake -Bbuild && cmake --build build
```
5. Test
```
cd build && ctest
```

## Example input file
We provide some example input files in config directory.
For example, the bert.cfg:
```ini
save_log_file=2
store_whole_block=1
mem_size=131072
log_directory=log/log_fuse_m131072_s1024_hi64_hn12_bs16_h1_b1_bert_2_genetic
hid_size=64
head_num=12
seq_length=1024
batch_size=16
head_blocksize=1
batch_blocksize=1
dim_order_opt=genetic
enable_compute_utilization_constraint=1
```

## Example usage
```
./build/DAT --config config/bert.cfg
```
It may take minutes to hours, depending on the computer's performance.

## Example output
```
A[n(1024,32,32)][d(64,4,16)][bsc(16,16,1)][hsc(12,12,1)] = S[n(1024,32,32)][m(1024,1,1024)][bsc(16,16,1)][hsc(12,12,1)] * V[m(1024,1,1024)][d(64,4,16)][bsc(16,16,1)][hsc(12,12,1)]
S[n(1024,32,32)][m(1024,1,1024)][bsc(16,16,1)][hsc(12,12,1)] = Q[n(1024,32,32)][q(64,4,16)][bsc(16,16,1)][hsc(12,12,1)] * K[m(1024,1,1024)][q(64,4,16)][bsc(16,16,1)][hsc(12,12,1)]
access times: 75264
access volume: 50331648
SRAM footprint: 98816
K[m(1024,4,256)][q(64,1,64)][bsc(16,16,1)][hsc(12,2,6)] = I2[m(1024,4,256)][k(768,48,16)][bsc(16,16,1)] * Wk[k(768,48,16)][q(64,1,64)][hsc(12,2,6)]
access times: 12544
access volume: 75497472
SRAM footprint: 108544
V[m(1024,4,256)][d(64,1,64)][bsc(16,16,1)][hsc(12,2,6)] = I3[m(1024,4,256)][p(768,48,16)][bsc(16,16,1)] * Wv[p(768,48,16)][d(64,1,64)][hsc(12,2,6)]
access times: 12544
access volume: 75497472
SRAM footprint: 108544
Q[n(1024,4,256)][q(64,1,64)][bsc(16,16,1)][hsc(12,2,6)] = I1[n(1024,4,256)][l(768,48,16)][bsc(16,16,1)] * Wq[l(768,48,16)][q(64,1,64)][hsc(12,2,6)]
access times: 12544
access volume: 75497472
SRAM footprint: 108544
-----------------------------------------
total access volume: 276824064
```


## Example log file
```csv
id,seq length,hid size,head num,dim size,mem size,fuse pattern,fuse num,fuse status,sub fuse num,sub fuse statuses,execute ranks,dim orders,dim block sizes,expand dims,change dim,operator expand dims,operator station tensor,operator dim block sizes,mem footprint,ops num,op chain mem access volume,mem access volume, compute time
1,1024,64,12,(bsc:16)(d:64)(hsc:12)(k:768)(l:768)(m:1024)(n:1024)(p:768)(q:64),131072,-1,1,(K:0)(Q:0)(S:1)(V:0),2,[(A:0)(K:1)(Q:0)(V:1)][(I2:0)(K:0)(Wk:0)][(I3:0)(V:0)(Wv:0)][(I1:0)(Q:0)(Wq:0)],[(0)(1)][(0)][(0)][(0)],{[(d)(n)(hsc)(bsc)(m)][(q)(n)(hsc)(bsc)(m)]}{[(k)(q)(hsc)(bsc)(m)]}{[(p)(m)(hsc)(bsc)(d)]}{[(l)(n)(hsc)(bsc)(q)]},[(mul_a_bsc:1)(mul_a_d:16)(mul_a_hsc:1)(mul_a_m:1024)(mul_a_n:32)(mul_s_bsc:1)(mul_s_hsc:1)(mul_s_m:1024)(mul_s_n:32)(mul_s_q:16)][(mul_k_bsc:1)(mul_k_hsc:6)(mul_k_k:16)(mul_k_m:256)(mul_k_q:64)][(mul_v_bsc:1)(mul_v_d:64)(mul_v_hsc:6)(mul_v_m:256)(mul_v_p:16)][(mul_q_bsc:1)(mul_q_hsc:6)(mul_q_l:16)(mul_q_n:256)(mul_q_q:64)],{[mul_a_A:][mul_a_S:][mul_a_V:(d)][mul_s_K:(q)][mul_s_Q:][mul_s_S:]}{[mul_k_I2:][mul_k_K:][mul_k_Wk:]}{[mul_v_I3:][mul_v_V:][mul_v_Wv:]}{[mul_q_I1:][mul_q_Q:][mul_q_Wq:]},[mul_a(d)][mul_s(q)][mul_k(k)][mul_v(p)][mul_q(l)],{[mul_a_V:(N)][mul_s_K:(K)]}{}{}{},[mul_a(S)][mul_s(S)][mul_k(K)][mul_v(V)][mul_q(Q)],[mul_a(B:1)(K:1024)(M:32)(N:16)][mul_s(B:1)(K:16)(M:32)(N:1024)][mul_k(B:1)(K:16)(M:256)(N:384)][mul_v(B:1)(K:16)(M:256)(N:384)][mul_q(B:1)(K:16)(M:256)(N:384)],108544,54760833024,(50331648)(75497472)(75497472)(75497472),276824064,55271424
```

## How to cite?
