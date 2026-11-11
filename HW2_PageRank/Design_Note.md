# HW2 Design Note - Parallel PageRank

**작성:** 2026-11-11 (코드 구현 전)  
**AI 활용:** ChatGPT (GPT-6) 개념 설명 및 설계 초안 정리

**1. 파일 읽기 및 블록 분할**  
전체 N개 목적지 노드를 p개 Rank에 연속해서 나눈다. `q=N/p`, `rem=N%p`일 때 Rank k의 노드 수는 `q+(k<rem?1:0)`, 시작 ID는 `k*q+min(k,rem)`이다. 각 Rank는 PRCSR1의 담당 `row_ptr` 및 in-edge `col` 구간만 읽고, `outdeg`와 이전 PageRank(OLD)는 전체 값을 사용한다. 간선 오프셋은 `int64_t`로 처리한다. 노드 수가 같아도 in-edge 수가 달라 계산량은 불균형할 수 있다.

**2. Dangling Node 처리**  
초기값은 모든 노드에서 `1/N`이고, 초기 Dangling Mass는 `D/N`이다. Outdegree가 0인 노드들의 점수를 모아 모든 노드에 `d × DanglingMass/N`만큼 분배한다. 다음 반복에 사용할 Dangling Mass는 이번에 계산한 NEW 값을 기준으로 구한다.

**3. Collective 통신 순서**  
Rank 0이 확인한 설정과 헤더를 `MPI_Bcast`로 공유한다. 반복마다 각 Rank가 담당 노드의 NEW, Local Residual, 다음 Dangling Mass를 계산한다. `MPI_Allreduce(MPI_SUM)`로 두 합계를 구하고 `MPI_Allgatherv`로 전체 NEW 벡터를 복원한 후 다음 반복에 사용한다. 모든 Rank의 호출 순서를 동일하게 유지한다.

**4. 수렴 및 성능 측정**  
정상 모드는 전체 L1 Residual이 `eps` 미만이면 종료하고, 성능 실험은 Benchmark Mode로 정확히 20회 수행한다. 파일 읽기는 Load로 따로 기록한다. 반복 직전 `MPI_Barrier` 후 `MPI_Wtime`으로 측정하며 Total, Compute, Allreduce, Allgatherv, Comm의 Rank별 최대 시간을 보고한다. AWS 1·2·4노드에서 각각 3회 이상 측정하여 중앙값과 최솟값·최댓값을 비교한다.
