#!/usr/bin/env bash
# Tổng hợp nhanh: tiến trình + kết quả đã có + còn thiếu + gợi ý lệnh tiếp theo.
cd /workspace/Recommendation-Machine-Unlearning 2>/dev/null || cd "$(dirname "$0")/.."

echo "########## [1] TIẾN TRÌNH ##########"
ps -eo pid,etime,cmd | grep -E "run_compare_all|code/method_[0-9]" | grep -v grep || echo "  (KHÔNG có job nào chạy)"
echo ""
echo "--- 5 dòng cuối orchestrator ---"
tail -5 logs/full.out 2>/dev/null || echo "  (chưa có logs/full.out)"
echo ""

echo "########## [2] TỔNG HỢP KẾT QUẢ ##########"
bash code/run_compare_all.sh --grep
echo ""

echo "########## [3] CÒN THIẾU ##########"
LIST="cmp1_ml-1m_full cmp1_ml-1m_ours cmp1_ml-1m_sisa cmp1_ml-1m_rec_ubp_author
cmp1_yelp2018_full cmp1_yelp2018_ours cmp1_yelp2018_sisa cmp1_yelp2018_rec_ubp_author
cmp2_ml-1m_signature cmp2_ml-1m_randomfrozen cmp2_ml-1m_ubpfrozen_embwmf cmp2_ml-1m_ubpfrozen_embbprfull cmp2_ml-1m_inpfrozen_embwmf cmp2_ml-1m_inpfrozen_embbprfull
cmp2_yelp2018_signature cmp2_yelp2018_randomfrozen cmp2_yelp2018_ubpfrozen_embwmf cmp2_yelp2018_ubpfrozen_embbprfull cmp2_yelp2018_inpfrozen_embwmf cmp2_yelp2018_inpfrozen_embbprfull
cmp3_ml-1m_pt1_embwmf cmp3_ml-1m_pt1_embbprfull cmp3_ml-1m_pt2_embwmf cmp3_ml-1m_pt2_embbprfull cmp3_ml-1m_pt3_random
cmp3_yelp2018_pt1_embwmf cmp3_yelp2018_pt1_embbprfull cmp3_yelp2018_pt2_embwmf cmp3_yelp2018_pt2_embbprfull cmp3_yelp2018_pt3_random
cmp4_ex6_ml-1m cmp4_ex6_yelp2018
cmp5_k4 cmp5_k8 cmp5_k16 cmp5_r0.001 cmp5_r0.005 cmp5_r0.01 cmp5_r0.02 cmp5_mode_random cmp5_mode_most cmp5_mode_fewest cmp5_sigd32 cmp5_sigd64 cmp5_sigd128
cmp6_ml-1m_ubp_mean cmp6_ml-1m_ubp_attn_lr0.05_ep60 cmp6_ml-1m_ubp_attn_lr0.005_ep60 cmp6_ml-1m_ubp_attn_lr0.005_ep200 cmp6_ml-1m_ubp_attn_lr0.001_ep200
cmp6_yelp2018_ubp_mean cmp6_yelp2018_ubp_attn_lr0.05_ep60 cmp6_yelp2018_ubp_attn_lr0.005_ep60 cmp6_yelp2018_ubp_attn_lr0.005_ep200 cmp6_yelp2018_ubp_attn_lr0.001_ep200"

ndone=0; nmissing=0; nrunning=0
for name in $LIST; do
  f="logs/${name}.log"
  if [ ! -f "$f" ]; then
    printf "  MISSING  %s\n" "$name"; nmissing=$((nmissing+1))
  elif grep -q "After (RETAINED" "$f" 2>/dev/null; then
    ndone=$((ndone+1))
  else
    printf "  RUNNING  %s\n" "$name"; nrunning=$((nrunning+1))
  fi
done
echo ""
echo "  DONE     : $ndone"
echo "  RUNNING  : $nrunning"
echo "  MISSING  : $nmissing"
echo "  TỔNG     : 57"
echo ""

echo "########## [4] GỢI Ý LỆNH TIẾP THEO ##########"
if [ "$nrunning" -gt 0 ]; then
  echo "  Job ĐANG CHẠY -> chờ. Theo dõi:"
  echo "    watch -n 60 'tail -3 logs/full.out'"
elif [ "$nmissing" -gt 0 ]; then
  echo "  Job ĐÃ DỪNG, còn $nmissing log chưa chạy. Chạy tiếp:"
  echo "    cd /workspace/Recommendation-Machine-Unlearning"
  echo "    git pull"
  echo "    nohup bash -c 'BLOCKS=\"1 2 3 4 5 6\" bash code/run_compare_all.sh' > logs/full.out 2>&1 &"
  echo "    tail -f logs/full.out"
else
  echo "  ĐÃ ĐỦ TẤT CẢ 57 log. In số cuối cùng:"
  echo "    bash code/run_compare_all.sh --grep"
  echo "  -> gửi output cho Claude để điền báo cáo."
fi
echo ""
echo "########## [5] CACHE ##########"
ls -la data/*/C*_type-*_num-8.pk data/*/wmf_embeddings.npz data/*/ours_partition_*.npz 2>/dev/null | head -20 || echo "  (chưa có cache nào)"
