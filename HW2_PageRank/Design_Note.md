# HW2 Design Note - Parallel PageRank

**작성일:** 2026-11-11 (본격적인 코드 구현 전)  
**참고:** HW2_PageRank.pdf  
**AI 활용:** ChatGPT (GPT-6)로 개념을 학습하고 설계 내용을 정리함.

**1. 파일 읽기와 노드 분할**  
전체 N개 노드를 p개 Rank에 연속된 구간으로 최대한 균등하게 나눈다. `q=N/p`, `rem=N%p`일 때 Rank k는 `n_k=q+(k<rem?1:0)`개를 맡고, 시작 노드는 `s_k=k*q+min(k,rem)`이다. 각 Rank는 PRCSR1 파일에서 자신의 노드 구간에 해당하는 `row_ptr`과 in-edge `col`만 읽는다. 다만 `outdeg`와 이전 PageRank 값(OLD)은 전체 노드의 값이 필요하다. 간선 수가 32비트 범위를 넘으므로 간선 위치와 파일 오프셋은 64비트 정수를 사용한다. 노드 수가 같아도 들어오는 간선 수가 달라 계산량 불균형이 생길 수 있다.

**2. Dangling Node 처리**  
처음에는 모든 노드의 PageRank를 `1/N`으로 설정한다. 나가는 간선이 없는 노드(outdegree=0)의 점수는 사라지지 않도록 전체 노드에 균등하게 분배한다. 처음 Dangling Mass는 `D/N`이며, 매 반복에서 새로 계산된 점수 중 Dangling Node의 합을 모아 다음 반복에 사용한다.

**3. MPI Collective 통신 순서**  
Rank 0이 검증한 실행 설정과 그래프 헤더를 `MPI_Bcast`로 공유한다. 각 반복에서는 담당 노드의 NEW 점수와 Local Residual, 다음 Dangling Mass를 계산한다. `MPI_Allreduce(MPI_SUM)`로 두 합계를 전체 Rank에 공유하고, `MPI_Allgatherv`로 각 Rank의 NEW 구간을 모아 전체 벡터를 만든다. 모든 Rank의 Collective 호출 순서는 같게 유지하며, Point-to-Point 통신은 사용하지 않는다.

**4. 수렴과 시간 측정**  
일반 모드는 전체 L1 Residual이 `eps`보다 작으면 종료하고, 그렇지 않으면 `max_iters`까지 실행한다. 성능 비교에서는 조건을 맞추기 위해 Benchmark Mode로 정확히 20회 계산한다. 파일 로딩 시간(Load)은 따로 기록하고, 반복 시작 직전 `MPI_Barrier` 후 `MPI_Wtime`으로 측정한다. Total, Compute, Allreduce, Allgatherv, Comm을 구분하고 Rank별 시간의 최대값을 보고한다. AWS 1/2/4노드에서 같은 조건으로 3회 이상 측정하여 중앙값과 최소·최대값을 비교할 계획이다.
