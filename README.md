Source code for our DAC 2024 paper, "Enabling Multiple Tensor-wise Operator Fusion for Transformer Models on Spatial Accelerators"

## Setup
1. Follow the instruction in gurobi website to get the gurobi licence 
2. Install c++ boost library.
3. Clone the repository.
```
$ git clone git@github.com:lxu28973/DAT.git
```
4. Compile the code.
```
$ cd DAT && cmake -Bbuild && cmake --build build
```
5. Test
```
$ cd build && ctest
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

## Example output file
```csv
id,seq length,hid size,head num,dim size,mem size,fuse pattern,fuse num,fuse status,sub fuse num,sub fuse statuses,execute ranks,dim orders,dim block sizes,expand dims,change dim,operator expand dims,operator station tensor,operator dim block sizes,mem footprint,ops num,op chain mem access volume,mem access volume, compute time
1,256,64,16,(bsc:16)(d:64)(hsc:16)(k:1024)(l:1024)(m:256)(n:256)(p:1024)(q:64),131072,-1,1,(K:0)(Q:0)(S:1)(V:0),0,[(A:0)(K:0)(Q:0)(V:0)][(I2:0)(K:0)(Wk:0)][(I1:0)(Q:0)(Wq:0)][(I3:0)(V:0)(Wv:0)],[(1)(0)][(0)][(0)][(0)],{[(q)(n)(m)(hsc)(bsc)][(n)(m)(d)(hsc)(bsc)]}{[(k)(hsc)(q)(m)(bsc)]}{[(l)(hsc)(n)(q)(bsc)]}{[(p)(hsc)(m)(d)(bsc)]},[(mul_a_bsc:1)(mul_a_d:16)(mul_a_hsc:1)(mul_a_m:256)(mul_a_n:16)(mul_s_bsc:1)(mul_s_hsc:1)(mul_s_m:256)(mul_s_n:256)(mul_s_q:16)][(mul_k_bsc:1)(mul_k_hsc:4)(mul_k_k:16)(mul_k_m:256)(mul_k_q:64)][(mul_q_bsc:1)(mul_q_hsc:4)(mul_q_l:16)(mul_q_n:256)(mul_q_q:64)][(mul_v_bsc:1)(mul_v_d:64)(mul_v_hsc:4)(mul_v_m:256)(mul_v_p:16)],{[mul_a_A:][mul_a_S:(n)(m)][mul_a_V:][mul_s_K:][mul_s_Q:][mul_s_S:(n)(m)]}{[mul_k_I2:][mul_k_K:][mul_k_Wk:]}{[mul_q_I1:][mul_q_Q:][mul_q_Wq:]}{[mul_v_I3:][mul_v_V:][mul_v_Wv:]},[mul_s(q)][mul_a(n)][mul_k(k)][mul_q(l)][mul_v(p)],{[mul_a_S:(K)(M)][mul_s_S:(M)(N)]}{}{}{},[mul_s(S)][mul_a(V)][mul_k(K)][mul_q(Q)][mul_v(V)],[mul_q(B:1)(K:16)(M:256)(N:256)][mul_v(B:1)(K:16)(M:256)(N:256)][mul_k(B:1)(K:16)(M:256)(N:256)][mul_s(B:1)(K:16)(M:256)(N:256)][mul_a(B:1)(K:256)(M:16)(N:16)],78080,15032385536,(16777216)(37748736)(37748736)(37748736),130023424,16392192
```

## How to cite?