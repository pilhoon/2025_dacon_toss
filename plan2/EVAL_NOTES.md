## Composite Metric
- Public scoring: 0.5*AP + 0.5*(1/(1+WLL))
- Track both AP and WLL; prioritize AP gains without exploding WLL.

## Checklists per run
- 데이터 통계 스냅샷 저장(mean/std, class ratio)
- 예측 분포 통계(mean/std/max, 히스토그램)
- 캘리브레이션 플롯(Reliability)
- 세그먼트 성능(AP by inventory_id, hour, user cohorts)


