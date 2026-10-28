# HW1 Design Note — 2D Heat Diffusion (MPI_Send/MPI_Recv)

> **Status:** Pre-implementation draft (before C code generation)  
> **Date:** 2026-10-28  
> **Basis:** HW1_2D_Heat_Diffusion_2026.pdf, lectures 5–8  
> **AI assistance:** ChatGPT (OpenAI), GPT-6 — design discussion and document drafting. The student must review and explain the design independently.

## 1. Domain Decomposition and Local Data

전역 `NX × NY` 격자를 `PX × PY`의 균등한 2D block으로 분할한다(`NX % PX = NY % PY = 0`). 각 rank의 process-grid 좌표는 `px = rank % PX`, `py = rank / PX`(정수 나눗셈)이며, local interior는 `localNX = NX/PX`, `localNY = NY/PY`이다. 각 rank는 1-cell halo를 포함한 `(localNY+2) × (localNX+2)` 크기의 row-major `double` 배열 두 개(`T_old`, `T_new`)를 소유한다. Interior의 local 좌표는 `x=1..localNX`, `y=1..localNY`이다. 같은 timestep에서는 `T_old`만 읽고 `T_new`에 기록한다.

## 2. Neighbor Rank and Boundary

이웃은 유효한 좌표에서만 `Left = rank-1` (`px>0`), `Right = rank+1` (`px<PX-1`), `Up = rank-PX` (`py>0`), `Down = rank+PX` (`py<PY-1`)로 구한다. 전역 경계에는 이웃 rank가 없으므로, 해당 외부 halo에 인접한 `T_old` interior 값을 복사해 Neumann (zero-flux) 조건을 적용한다.

## 3. Send and Receive Buffers

- **Up/Down:** 첫/마지막 interior **row**를 `localNX`개 `MPI_DOUBLE` 원소로 직접 전송하고 위/아래 halo row에 직접 수신한다(연속 메모리).
- **Left/Right:** 첫/마지막 interior **column**의 `localNY`개 원소를 임시 send buffer로 **pack**해 전송한다. 임시 receive buffer에 수신한 후 대응하는 halo column으로 **unpack**한다(비연속 메모리).
- 전송 방향을 표시하는 tag는 `LEFT=10`, `RIGHT=11`, `UP=12`, `DOWN=13`으로 구분한다. 예를 들어 오른쪽 이웃에서 오는 데이터는 상대방의 `LEFT` tag로 수신한다.

## 4. Blocking Communication / Deadlock Prevention

모든 rank가 이웃이 존재하는 방향만 **Left → Right → Up → Down** 순서로 처리한다. **Left/Up:** `Recv → Send`; **Right/Down:** `Send → Recv`. 수신 시에는 이웃이 보내는 방향의 tag를 사용한다(Left 수신: RIGHT, Right 수신: LEFT, Up 수신: DOWN, Down 수신: UP). 이 순서는 왼쪽/위쪽 끝 rank부터 대응하는 Send/Recv를 진행시키므로 MPI 내부 buffering에 의존하지 않는다. Halo 교환에는 **blocking `MPI_Send`/`MPI_Recv`만** 사용하며, collective/`MPI_Sendrecv`/nonblocking 통신은 사용하지 않는다.

---

**Note:** 이 파일은 과제 진행 및 AI 사용 이력 기록을 위한 **구현 전 설계 초안**이다. 최종 보고서의 Design Note는 0.5페이지 이내로 편집하고, 구현·실측 결과와 일치하는지 재확인한다.
