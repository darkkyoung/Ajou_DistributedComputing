# HW2 Parallel PageRank — V2 (PRCSR1 file mode)

현재 구현 범위: V1 Ring/Star, V2 PRCSR1 file input, Rank별 로컬 in-edge CSR 및 전체 outdeg 로딩, dangling mass 보정, MPI collectives, 전체 벡터 dump. **실제 AWS uk-2007-05 실험은 아직 미검증**입니다.

## 1. Ubuntu / Open MPI에서 빌드

```bash
cd ~/workspace/Ajou_DistributedComputing
git switch work/hw2-pagerank
git pull --ff-only
cd HW2_PageRank
make
```

필요 패키지: `build-essential openmpi-bin libopenmpi-dev python3`. 빌드 옵션: `mpicc -O2 -std=c11 -Wall -Wextra -Wpedantic -lm`.

## 2. V1 Ring / Star 회귀 테스트

```bash
mpirun -np 4 ./pagerank ring 8 0.85 1e-12 500 8 0
mpirun -np 4 ./pagerank star 4 0.85 1e-12 500 4 0
```

정상: Ring 각 노드 0.125, Star 중심 약 0.4797297297 / 주변 0.1734234234.

## 3. V2 로컬 파일 모드: N=5, M=5, Dangling=1

테스트 그래프: `0→{1,2}`, `1→2`, `2→0`, `3→2`, `4→(없음)`. `N % 4 != 0`이며 Dangling Node 보정과 비균등 in-edge 분할을 함께 검증합니다.

```bash
mkdir -p results
python3 tools/tiny_prcsr_test.py create results/tiny.prcsr1
mpirun -np 1 ./pagerank file results/tiny.prcsr1 0.85 1e-12 500 5 20 results/tiny_np1.f64 | tee results/tiny_np1.txt
mpirun -np 4 ./pagerank file results/tiny.prcsr1 0.85 1e-12 500 5 20 results/tiny_np4.f64 | tee results/tiny_np4.txt
python3 tools/tiny_prcsr_test.py check results/tiny_np1.f64 results/tiny_np4.f64
```

`check`에서 **PASS 2회 + cross-rank PASS**를 확인해야 합니다. 로컬 테스트가 통과해도 전체 uk-2007-05 정확성이 증명된 것은 아닙니다.

### 오류 처리 검사

```bash
mpirun -np 4 ./pagerank file results/does-not-exist.bin 0.85 1e-12 500
```

정상: 오류 메시지 후 모든 Rank가 종료되어야 함(비정상 종료 코드 반환은 의도됨).

### 추가 경계 검사 (선택)

```bash
mpirun --oversubscribe -np 8 ./pagerank file results/tiny.prcsr1 0.85 1e-12 500 5 20 results/tiny_np8.f64
python3 tools/tiny_prcsr_test.py check results/tiny_np8.f64
```

## 4. 실제 AWS 데이터 검증(향후 실행할 것)

- 각 AWS 노드의 로컬 `/data/hw2`에 제공 S3의 `uk-2007-05.bin`과 기준 벡터·도구 다운로드 후 SHA256 검증.
- 공용 EFS `/shared/hw2`에는 소스·실행 파일·결과 로그만 배치.
- 파일 모드는 `file /data/hw2/uk-2007-05.bin`으로 실행.
- **성능:** 1/2/4 AWS 노드 각각 4/8/16 Rank, `0.85 1e-10 500 10 20`, 각 3회 이상.
- **정확성:** 1노드 4 Rank와 4노드 16 Rank에서 `dump_path` 지정, 배포 `tools/compare_vectors.py`로 전체 L1 및 최대 절대 차이 검사. 허용오차는 배포 `README.txt` 기준.
- `[Result] Total`은 반복 구간만, `Load`는 파일 읽기 및 초기화 시간을 따로 기록.
- `/usr/bin/time -v` 또는 `/proc/<PID>/status` VmHWM으로 실제 메모리 최대치 확인.

## 5. AI 사용 기록

- 2026-11-11: ChatGPT GPT-6와 사용자가 그래프 분할, dangling mass, collectives, 종료 조건을 질의응답해 코드 생성 전에 `Design_Note.md` 작성.
- 2026-11-11: ChatGPT GPT-6가 V1 Ring/Star MPI 프로그램 초안 작성. 사용자 로컬에서 Ring 1/4 Rank, Star 1/4 Rank 검증 통과 및 로그 업로드.
- 2026-11-13: ChatGPT GPT-6가 기존 V1에 PRCSR1 헤더/파일 크기 검사, 파일 내 로컬 row_ptr/col 읽기, 64비트 간선/오프셋, outdeg 검증, 실패 시 집단 종료를 추가. 작은 파일 테스트 스크립트를 추가. **V2는 사용자가 로컬 MPI 환경에서 실행하여 검증 예정.**
- 실제 AWS 결과와 SHA256·메모리·성능 지표는 측정 전이며 만들어내지 않음.
