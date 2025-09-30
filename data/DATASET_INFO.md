download link: https://dacon.io/competitions/official/236575/data (8.3G)


Dataset Info.

train.parquet [파일] :
총 10,704,179개 샘플
총 119개 ('clicked' Target 컬럼 포함) 컬럼 존재
gender : 성별
age_group : 연령 그룹
inventory_id : 지면 ID
day_of_week : 주번호
hour : 시간
seq : 유저 서버 로그 시퀀스
l_feat_* : 속성 정보 피처 (l_feat_14는 Ads set)
feat_e_* : 정보영역 e 피처
feat_d_* : 정보영역 d 피처
feat_c_* : 정보영역 c 피처
feat_b_* : 정보영역 b 피처
feat_a_* : 정보영역 a 피처
history_a_* : 과거 인기도 피처
clicked : 클릭 여부 (Label)


test.parquet [파일] :
총 1,527,298개 샘플
총 118개 ('ID' 식별자 컬럼 포함) 컬럼 존재
ID : 샘플 식별자
gender : 성별
age_group : 연령 그룹
inventory_id : 지면 ID
day_of_week : 주번호
hour : 시간
seq : 유저 서버 로그 시퀀스
l_feat_* : 속성 정보 피처 (l_feat_14는 Ads set)
feat_e_* : 정보영역 e 피처
feat_d_* : 정보영역 d 피처
feat_c_* : 정보영역 c 피처
feat_b_* : 정보영역 b 피처
feat_a_* : 정보영역 a 피처
history_a_* : 과거 인기도 피처


sample_submission.csv [파일] - 제출 양식
ID : 샘플 식별자
clicked : 광고를 클릭할 확률 (0 ~ 1)

