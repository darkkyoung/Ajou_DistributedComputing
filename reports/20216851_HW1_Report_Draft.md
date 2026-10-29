# HW1 — 2D Heat Diffusion using MPI_Send / MPI_Recv

> 과목: 분산병렬컴퓨팅 · 학번: 20216851 · 성명: [입력]
>
> **보고서 초안.** 실제 실행 화면과 원본 AI 대화 부록은 제출 전 삽입해야 합니다. Word 편집본에는 도식·수식 이미지와 성능 그래프가 포함되어 있습니다.

## 1. Design Note

*2026-10-28, 구현 전 설계 초안에 기반. 초안 작성에는 ChatGPT의 지원을 받음.*

**Local Data / Halo.** 전역 NX×NY 격자를 PX×PY개의 블록으로 균등 분할한다. Rank의 Interior 크기는 (NX/PX)×(NY/PY)이고, Halo를 포함한 (localNY+2)×(localNX+2) 크기의 T_old와 T_new를 사용한다.

**Neighbor Rank.** px=rank%PX, py=rank/PX로 좌표를 구한다. 실제 이웃이 존재할 때만 Left=rank−1, Right=rank+1, Up=rank−PX, Down=rank+PX를 사용한다.

**Send/Receive Buffer.** 상하 Row는 연속 메모리이므로 직접 송수신한다. 좌우 Column은 임시 버퍼에 Pack하여 보내고 수신한 값을 Halo Column에 Unpack한다.

**Blocking Communication.** Left→Right→Up→Down 순서로 처리한다. Left/Up은 Recv→Send, Right/Down은 Send→Recv로 맞추며 방향별 Tag로 메시지를 구분한다. 내부 MPI 버퍼링에 의존하지 않고 통신한다.

## 2. Implementation & MPI Communication

### 2.1 Domain Decomposition

각 Rank는 자신의 Interior를 계산하며, 이웃 온도는 1-cell Halo로 참조한다. rank=py×PX+px로 매핑한다. 2×2 Process Grid의 Rank 배치는 아래와 같다.

| Rank 0 | Rank 1 |
|:--:|:--:|
| Rank 2 | Rank 3 |

전역 외곽과 프로세스 사이 내부 경계를 구분한다.

### 2.2 Halo Exchange

MPI Rank는 독립적인 주소 공간을 사용하므로 이웃의 데이터를 직접 참조할 수 없다. 매 timestep에 경계 Row/Column을 교환하여 Halo를 갱신한다. Row-major 저장에서 Column은 연속되지 않으므로 sendColumn/recvColumn 버퍼를 사용한다.

오른쪽 방향 교환의 핵심 흐름:

~~~c
for (int y = 1; y <= localNY; y++)
    sendColumn[y - 1] = T_old[(size_t)y * width + localNX];
MPI_Send(sendColumn, localNY, MPI_DOUBLE, right, TAG_RIGHT, MPI_COMM_WORLD);
MPI_Recv(recvColumn, localNY, MPI_DOUBLE, right, TAG_LEFT,
         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
for (int y = 1; y <= localNY; y++)
    T_old[(size_t)y * width + localNX + 1] = recvColumn[y - 1];
~~~

상하 Row는 배열 주소를 직접 전달한다. Left/Up은 Recv→Send, Right/Down은 Send→Recv로 구성해 모든 Rank가 Send 후 대기하는 상황을 피한다.

### 2.3 Stencil, Neumann Boundary & Double Buffer

5-point Stencil은 자기 셀 C와 상하좌우 N/S/W/E의 이전 온도로 새 온도를 계산한다.

$$T_{\mathrm{new}}=C+\alpha(N+S+W+E-4C),\qquad 0<\alpha\leq0.25$$

같은 timestep에서 새 값을 덮어쓰면 계산 순서에 따라 결과가 달라질 수 있으므로 T_old에서 읽고 T_new에 기록한 뒤 포인터를 교환한다. 전역 격자 밖 Halo에는 인접 Interior 값을 복사해 Neumann(Zero Heat Flux) 경계를 적용하고, Rank 사이 Halo에는 실제 이웃 Rank가 전송한 값을 사용한다.

### 2.4 Global Statistics & Timing

Rank별 Interior의 Local Sum/Max와 경과시간을 Rank 0이 MPI_Send/Recv로 수집한다. Global Sum, Global Max 및 Rank별 Simulation Loop 경과시간의 최댓값을 출력한다. MPI_Wtime 측정에는 경계 설정, Halo Exchange 및 Stencil 계산을 포함하되 초기화와 종료 후 통계 수집은 제외한다. MPI_Reduce/MPI_Barrier를 사용하지 않는다.

## 3. Build & Run

| 항목 | 환경 |
|---|---|
| CPU | AMD Ryzen 9 7900X · 12 physical / 24 logical cores |
| OS | WSL2 Ubuntu |
| MPI | Open MPI 4.1.6 |
| Compiler | GCC 13.3.0 |
| 옵션 | -O2 -std=c11 -Wall -Wextra |

~~~bash
mpicc -O2 -std=c11 -Wall -Wextra -o heat_diffusion 20216851_HW1_heat_diffusion.c
mpirun -np 4 ./heat_diffusion 4 4 2 2 4 0.20
mpirun -np 4 ./heat_diffusion 512 512 2 2 200 0.20
mpirun -np 4 ./heat_diffusion 2048 2048 2 2 300 0.20
~~~

성능 실험은 동일 머신·옵션에서 설정별 Warm-up 1회, 측정 3회 실시하고 중앙값(Median)을 사용했다.

## 4. Experimental Results

### 4.1 Correctness Verification

초기 상태는 전역 (0,0)의 온도만 100, 나머지는 0이며 alpha=0.20을 사용했다. 같은 격자 크기에서 1×1을 기준으로 비교했다.

**Table 1. Correctness**

| NX | NY | Steps | α | Grid | Sum | Max | 기준 대비 차이 (Sum/Max) |
|---:|---:|---:|---:|---|---:|---:|---|
| 4 | 4 | 4 | 0.20 | 1×1 | 100.000000 | 28.320000 | 기준 |
| 4 | 4 | 4 | 0.20 | 2×2 | 100.000000 | 28.320000 | 0 / 0 |
| 512 | 512 | 200 | 0.20 | 1×1 | 99.999999999999 | 0.789361906041 | 기준 |
| 512 | 512 | 200 | 0.20 | 2×2 | 99.999999999999 | 0.789361906041 | 0 / 0 |

**Table 2. 4×4 최종 온도 분포 — 1×1 및 2×2가 모두 아래 값과 일치**

| y＼x | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 28.32 | 17.60 | 6.72 | 1.60 |
| 1 | 17.60 | 10.56 | 3.52 | 0.64 |
| 2 | 6.72 | 3.52 | 0.96 | 0.00 |
| 3 | 1.60 | 0.64 | 0.00 | 0.00 |

4×4에서는 실제 열이 Rank 경계를 넘으므로 Halo Exchange를 직접 검증할 수 있다. 반면 512×512, 200 steps는 열이 다른 Rank까지 도달하지 않아 큰 격자에서의 계산·열 보존 및 Sum/Max 비교를 확인하는 보완 실험이다.

> **[제출 전 실제 실행 화면 삽입: 4×4의 1×1/2×2 터미널 결과]**

### 4.2 Strong Scaling & Process-grid Comparison

NX=NY=2048, steps=300, alpha=0.20. 각 Rank의 Simulation Loop 최대 시간을 측정했고 설정당 3회 중앙값을 사용했다.

$$S_p=\frac{T_1}{T_p},\qquad E_p=\frac{S_p}{p}$$

**Table 3. Performance**

| Procs | Grid | Local size | Halo / Msgs per step | Median (s) | Speedup / Efficiency | Sum / Max |
|---:|---|---|---|---:|---|---|
| 1 | 1×1 | 2048×2048 | 0 / 0 | 1.227150 | 1.000 / 100.0% | 100.000000 / 0.527659 |
| 2 | 1×2 | 2048×1024 | 4096 / 2 | 0.698465 | 1.757 / 87.8% | 100.000000 / 0.527659 |
| 4 | 2×2 | 1024×1024 | 8192 / 8 | 0.463206 | 2.649 / 66.2% | 100.000000 / 0.527659 |
| 4 | 1×4 | 2048×512 | 12288 / 6 | 0.453273 | 2.707 / 67.7% | 100.000000 / 0.527659 |
| 4 | 4×1 | 512×2048 | 12288 / 6 | 0.404819 | 3.031 / 75.8% | 100.000000 / 0.527659 |

Sum/Max는 공간상 여섯째 자리까지 표시했다. 원본 결과는 모든 설정에서 Sum=99.999999999999, Max=0.527658552559였다. Halo는 전체 송신 double 원소 수, Msgs는 MPI_Send 횟수이다.

프로세스 1→2→4개(1×1, 1×2, 2×2)의 Speedup은 1.000→1.757→2.649로 증가했지만 이상적인 4배에는 미치지 못했다. 통신·대기·메모리 시스템 등이 가능한 원인이나 개별 통신시간은 측정하지 않았다.

> **[제출 전 실제 실행 화면 삽입: 2048×2048 성능 측정 결과]**

### 4.3 Process-grid Shape Interpretation

각 Rank의 계산 면적은 1,048,576셀로 같다. 2×2는 내부 경계 길이 1024인 경계 네 개를 양방향 교환하여 Halo=4×1024×2=8192개이다. 1×4 및 4×1은 길이 2048인 경계 세 개여서 Halo=3×2048×2=12288개이다.

2×2는 데이터량이 가장 적지만 메시지는 8개이며, 나머지 두 구성은 6개이다. 실험에서는 4×1의 중앙값(0.404819초)이 2×2(0.463206초)보다 짧았다. 다만 4×1의 세 번 측정값은 0.477326, 0.404819, 0.396144초로 변동이 있어 일반적인 최적성을 단정할 수 없다.

## 5. Required Analysis Questions

### A. Halo Exchange가 필요한 이유

MPI Rank는 독립적인 주소 공간을 사용한다. Rank 1은 Rank 0이 보낸 마지막 Interior Column의 값을 자신의 Left Halo에 수신한다. 이 값은 같은 timestep Stencil의 서쪽 이웃(W)으로 사용되고 계산 결과는 다음 timestep T_new에 저장된다. 통신하지 않으면 프로세스 경계에서 실제 이웃 온도가 반영되지 않는다.

### B. Blocking Send-first와 Deadlock

모든 Rank가 MPI_Send를 먼저 호출하면 상대방의 Recv를 기다리며 데드락이 발생할 수 있다. 내부 버퍼링에 의존하지 않도록 Left/Up은 Recv→Send, Right/Down은 Send→Recv를 사용하고 방향별 Tag로 정확히 매칭한다.

### C. 4개 Rank에서 통신 비용이 다른 이유

계산 면적은 동일하지만 내부 경계 길이가 다르다. 2×2의 Halo는 8192개, 1×4/4×1은 12288개이다. 반면 메시지 수는 2×2가 8개, 나머지는 6개이므로 전송 데이터량만으로 실행시간을 설명할 수 없다.

### D. Strong Scaling이 이상적이지 않은 이유

프로세스 수가 늘면 계산량은 줄지만 Halo 전송, Blocking 대기, Column Pack/Unpack 등이 추가된다. 2개 Rank는 Speedup 1.757/Efficiency 87.8%, 4개 Rank의 2×2는 2.649/66.2%였다. 측정하지 않은 병목은 단정하지 않는다.

### E. 중요한 통신 코드 두 곳

**Horizontal Column Pack/Unpack.** 마지막 Interior Column을 sendColumn에 모아 TAG_RIGHT로 보내고, 이웃이 TAG_LEFT로 보낸 값을 받아 Right Halo에 복사한다. Tag나 Halo 위치가 잘못되면 메시지가 매칭되지 않거나 틀린 이웃 온도를 사용한다.

**Vertical Row Send/Recv.** 위쪽 이웃의 Row는 상단 Halo인 &T_old[1]에 localNX개를 직접 받고, 첫 Interior Row인 &T_old[width+1]에서 송신한다. Row는 연속 메모리이므로 Pack이 필요 없다. 주소 또는 개수를 틀리면 다른 위치를 읽거나 덮어쓴다.

### F. AI 최초 코드와 최종 코드의 주요 수정

초기 V2의 AI 지원 코드는 단일 Rank 계산에서 네 방향에 Neumann 조건을 적용했다. 이를 다중 Rank에 그대로 적용하면 내부 프로세스 경계까지 단열 처리된다. V4~V6에서는 전역 외부에만 Neumann을 적용하고 내부는 Row/Column Halo Exchange로 변경했다. 이 확장 역시 AI 생성·수정 지원을 받았고, 4×4 16개 셀 및 512×512 통계 비교로 검증했다.

추가로 V7에서는 localElapsed가 timestep 블록 내부에 선언되어 외부 통계 수집 코드에서 참조되지 않는 Scope 오류를 발견했다. 계산을 반복문 종료 후로 이동하여 수정했다.

## 6. References

1. 아주대학교 분산병렬컴퓨팅, *HW1_2D_Heat_Diffusion_2026.pdf*, 2026-2.
2. 아주대학교 분산병렬컴퓨팅, 강의자료 5~8장 중 병렬 설계 및 MPI 통신 관련 내용.
3. [Ajou_DistributedComputing — work/hw1-my-implementation](https://github.com/darkkyoung/Ajou_DistributedComputing/tree/work/hw1-my-implementation): 소스 및 측정 자료.
4. [OpenAI ChatGPT](https://chatgpt.com/): AI 코드·보고서 작성 지원 도구.

## 7. Generative AI Usage

**Tool / Model / Dates:** ChatGPT (OpenAI), GPT-6, 2026-10-28~29.

과제 이해, Design Note 초안 작성, C 코드 단계별 생성·수정, 컴파일 오류 분석 및 Python 실험 스크립트 생성을 지원받았다. C 구현의 상당 부분에 AI 코드 지원이 있었음을 파일 상단에 기록했다. 중복 주석을 줄이고 **입력 검증·수평 Halo·수직 Halo·Global Statistics·4×4 재구성의 5개 핵심 영역**에 개별 AI 주석을 유지했다. 주석 정리로 코드 로직이나 실제 AI 사용 범위가 바뀐 것은 아니다.

| 단계 | 주요 AI 지원 영역 |
|---|---|
| V0–V3 | MPI 초기화, 입력 검증, Local Grid, Rank Mapping, Stencil, Double Buffer |
| V4–V6 | Row/Column Halo Exchange, Pack/Unpack, Blocking 통신 순서 |
| V7 | Global Sum/Max, MPI_Wtime, 4×4 결과 재구성 |
| V8 | Python 성능 실험 및 CSV 집계 |

디버깅에서는 V2의 px·py 관련 경고와 V7의 localElapsed Scope 오류를 수정했다. 최종 정리에서는 중복 AI 주석 6개를 제거했고, 계산 및 MPI 통신 로직은 변경하지 않았다.

보고서 초안 작성과 내용 구성, 문맥 연결, 맞춤법, 문장 배치, 가독성 개선, 수식과 표 구성에도 ChatGPT 지원을 받았다. 정확성과 성능은 실제 WSL2 실행 및 GitHub CSV 측정값에 근거해 점검했으며 AI 답변을 검증 근거로 삼지 않았다.

## Appendix A. AI Conversation Records

> **필수 부록 미완성:** 실제 사용자 질문과 AI 답변의 원문을 아래 구간에 삽입해야 합니다. 새로 재현한 대화를 당시 원본으로 제시하지 않습니다.

| 구간 | 사용자 질문 원문 | AI 답변 원문 |
|---|---|---|
| A-1. Design Note / V0–V3 | [실제 대화 원문] | [실제 대화 원문] |
| A-2. V4–V6 Halo Exchange | [실제 대화 원문] | [실제 대화 원문] |
| A-3. V7 수정 및 오류 해결 | [실제 대화 원문] | [실제 대화 원문] |
| A-4. V8 실험 설계 | [실제 대화 원문] | [실제 대화 원문] |

추가 원본 자료: results/v8_raw.csv, results/v8_summary.csv, results/v8_environment.txt.
