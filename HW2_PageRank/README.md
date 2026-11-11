# HW2 Parallel PageRank — V1 (Ring / Star)

현재 구현 범위는 **작은 합성 그래프(Ring/Star)** 의 MPI Collective PageRank 검증입니다. 실제 `uk-2007-05.bin` PRCSR1 파일 로딩은 다음 단계(V2)에서 구현합니다. `file` 입력은 **아직 지원하지 않습니다**.

- `20216851_HW2_pagerank.c`: V1 MPI PageRank 구현
- `Design_Note.md`: 본격 구현 전에 작성한 설계 노트
- `Makefile`: Open MPI C 컴파일 설정

## 빌드

```bash
cd ~/workspace/Ajou_DistributedComputing/HW2_PageRank
make
```

## 정확성 검증

```bash
mpirun -np 1 ./pagerank ring 8 0.85 1e-12 500 8 0
mpirun -np 4 ./pagerank ring 8 0.85 1e-12 500 8 0
mpirun -np 1 ./pagerank star 4 0.85 1e-12 500 4 0
mpirun -np 4 ./pagerank star 4 0.85 1e-12 500 4 0
```

- Ring: 8개 노드 각각 0.125, 초기 Residual 약 0.
- 양방향 Star: 중심(노드 0) 약 0.4797297297, 주변 노드 각각 약 0.1734234234.
- benchmark_iters > 0 이면 수렴 여부와 무관하게 정확히 지정된 횟수만 실행합니다.
- 노드 수보다 Rank 수가 많을 때(`-np 8 ./pagerank star 4 ...`)도 안전하게 동작해야 합니다.
- 출력: [Result], [Stats], [Balance], [Top].

## 작업 기록 및 AI 사용

- 2026-11-11: 학생이 PageRank 분할·Dangling·Collective·수렴 방식에 대해 ChatGPT(GPT-6)와 질의응답 후 Design Note를 작성했습니다.
- 2026-11-11: ChatGPT(GPT-6)가 위 설계에 맞춘 V1 C 코드를 초안으로 작성했습니다. 원격 실행 결과는 학생 로컬에서 직접 확인하고 기록해야 합니다.
- V2 계획: 제공된 load_slice_example.c 참고해 PRCSR1 헤더/CSR 부분 로딩, 대용량 성능 실험, 기준 벡터 정량 비교 추가.
