# my-car-astar

TurtleBot3(waffle_pi)가 **지도 위에서 A\* 알고리즘으로 길을 찾아 스스로 주행**하는 ROS2 Humble 노드입니다.
주행 중 앞에 **장애물이 나타나면 돌아가고**, 카메라에 **사람이 가까이 오면 비상정지**합니다.

| 파일 | 설명 |
|---|---|
| `astar_drive_yolo.py` | 주행용 노드 (A\* + Pure Pursuit + 라이다 장애물 회피 + YOLO 사람 비상정지) |
| `README.md` | 설명문 |

## 1. 이 노드가 하는 일

| 기능 | 설명 |
|---|---|
| 경로 계획 | RViz에서 찍은 목적지까지 **A\*** 알고리즘으로 최단 경로를 계산 |
| 경로 따라가기 | **Pure Pursuit**: 경로 위 앞쪽(0.5 m) 점을 향해 부드럽게 주행 |
| 장애물 회피 | **라이다**로 지도에 없던 장애물을 발견하면 피해서 경로를 다시 계산 |
| 비상정지 | **YOLO**가 카메라에서 사람을 찾고, 사람이 가까우면 즉시 정지 |

## 2. 동작 순서

```
RViz에서 2D Pose Estimate(출발 위치) → 2D Goal Pose(목적지)
        │
        ▼
 지도의 벽을 로봇 크기만큼 부풀림 → A*로 경로 계산 (/astar_path 로 RViz에 표시)
        │
        ▼
 Pure Pursuit로 경로를 따라 주행 (/cmd_vel)
        │
        ├─ 앞 0.35 m 안에 장애물 → 그 자리를 막힌 곳으로 표시하고 A* 재계산
        ├─ 사람이 0.5 m 안에 접근 → 즉시 정지 (사라지면 이어서 주행)
        └─ 목적지 0.12 m 안에 도착 → "목표 도착!" 출력 후 정지
```

사용하는 토픽: `/map`, `/amcl_pose`, `/goal_pose`, `/scan`(라이다), `/image_raw/compressed`(카메라), `/cmd_vel`(주행 명령), `/astar_path`(경로 표시)

## 3. 핵심 개념

- **A\* (에이스타):** 출발점에서 목적지까지 "지금까지 온 거리 + 남은 거리 짐작"이 가장 작은 쪽부터 찾아서 최단 경로를 구하는 알고리즘입니다. 이 코드는 8방향으로 움직이고, 벽 모서리를 긁는 대각선 이동은 막았습니다.
- **벽 부풀리기:** 로봇은 점이 아니라 크기가 있어서, 벽을 로봇 반지름(0.22 m)만큼 두껍게 만들어 두고 길을 찾습니다. 그래야 벽에 닿지 않습니다.
- **Pure Pursuit:** 로봇 앞쪽에 있는 경로 위의 점을 "토끼"처럼 정하고 계속 따라가는 방식입니다.
- **비상정지(YOLO + 라이다):** 카메라 화면에서 **사람이 보이는 방향**을 구한 뒤, **그 방향의 라이다 거리**가 0.5 m 이하일 때만 멈춥니다. 사람이 멀리 보이기만 할 때는 멈추지 않습니다.

## 4. 비상정지 상세

- 모델: 사전학습된 **`yolov8n.pt`**를 그대로 사용하고, **사람(person)만** 찾도록 제한했습니다.
- 사람이 `estop_conf`(0.7) 이상의 확신도로 보이고, 그 방향의 라이다 거리가 `estop_distance_threshold`(0.5 m) 이하이면 **즉시 정지**합니다.
- 사람이 사라지고 `estop_release_delay`(0.5초)가 지나면 **목표와 경로를 그대로 두고 이어서 주행**합니다.
- 카메라 영상이 `frame_timeout`(1초) 넘게 들어오지 않으면 안전을 위해 **정지**합니다. 영상이 돌아오면 이어서 주행합니다.
- YOLO 계산은 **별도 스레드**에서 하므로, YOLO가 느려도 주행과 라이다 수신이 멈추지 않습니다.
- 화면 창에는 사람 박스와 `person 0.85 0.43m`(확신도, 라이다 거리)가 표시되고, 정지 중에는 빨간 `EMERGENCY STOP` 문구가 뜹니다.
- 모델 파일이나 라이브러리가 없으면 **비상정지만 꺼지고** 주행은 그대로 동작합니다.

## 5. 조절할 수 있는 값 (`astar_drive_yolo.py`의 `__init__` 위쪽)

| 이름 | 기본값 | 의미 |
|---|---|---|
| `linear_speed` | `0.10` | 주행 속도 (m/s) |
| `lookahead` | `0.5` | 앞쪽 몇 m 지점을 따라갈지 |
| `robot_radius` | `0.22` | 벽 부풀리기 반지름 (m) |
| `obs_threshold` | `0.35` | 이 거리 안에 앞 장애물이 있으면 회피 (m) |
| `estop_enabled` | `True` | `False`면 비상정지 끄기 (카메라 없이 주행만 할 때) |
| `model_path` | `/home/student/my_ws/src/my_bot/models/yolov8n.pt` | YOLO 모델 파일 경로 (**자기 환경에 맞게 수정**) |
| `estop_conf` | `0.7` | 이 확신도 이상일 때만 사람으로 인정 |
| `estop_distance_threshold` | `0.5` | 사람이 이 거리(m)보다 가까우면 정지 |
| `estop_release_delay` | `0.5` | 사람이 사라진 뒤 출발까지 기다리는 시간(초) |
| `use_compressed` | `True` | `True`: `/image_raw/compressed` 사용, `False`: `/image_raw` 사용 |
| `show_window` | `True` | 검출 화면 창 표시 (느리면 `False`) |

## 6. 설치

```bash
sudo apt install python3-numpy ros-humble-cv-bridge ros-humble-v4l2-camera ros-humble-compressed-image-transport
pip install ultralytics
```
`ros-humble-compressed-image-transport`는 **로봇과 PC 양쪽**에 설치합니다.

**1) 코드와 모델 파일 복사**
```bash
cp astar_drive_yolo.py ~/my_ws/src/my_bot/my_bot/
mkdir -p ~/my_ws/src/my_bot/models
cp yolov8n.pt ~/my_ws/src/my_bot/models/      # 모델 파일은 저장소에 올리지 않았습니다
```
`model_path`의 경로와 사용자 이름이 다르면 코드의 `self.model_path`를 자기 경로로 바꿉니다.

**2) `setup.py`의 `entry_points`에 한 줄 추가**
```python
'console_scripts': [
    ...,
    'astar_drive_yolo = my_bot.astar_drive_yolo:main',
],
```

**3) `package.xml`에 추가**
```xml
<exec_depend>nav_msgs</exec_depend>
<exec_depend>cv_bridge</exec_depend>
```

**4) 빌드**
```bash
cd ~/my_ws && colcon build --packages-select my_bot && source install/setup.bash
```

## 7. 실행 순서

```bash
# (로봇) 브링업
ros2 launch turtlebot3_bringup robot.launch.py
# (로봇) 카메라 노드
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320,240]"
# (PC) 압축 영상이 오는지 확인
ros2 topic hz /image_raw/compressed
# (PC) 위치 추정 (지도 + AMCL + RViz)
ros2 launch my_bot tb3_localization_real.launch.py map:=/home/<사용자>/my_map.yaml
# (PC) 주행 노드
ros2 run my_bot astar_drive_yolo
```

RViz에서 **`2D Pose Estimate`**로 출발 위치를 찍고 **`2D Goal Pose`**로 목적지를 찍으면 출발합니다.
(`Nav2 Goal`이 아닙니다.) `Add → Path → Topic /astar_path`를 추가하면 계산된 경로를 볼 수 있습니다.

## 8. 주의사항

- 카메라 영상이 안 오면 로봇이 **일부러 정지**합니다. 로그에 `영상이 들어오지 않아 정지합니다`가 뜨면 카메라 노드와 토픽을 확인하세요.
- `/image_raw/compressed`가 없으면 `use_compressed = False`로 바꿔 `/image_raw`를 사용합니다. 와이파이로 원본 영상을 받으면 끊길 수 있습니다.
- 카메라가 로봇 **정면**을 보고 있다는 가정으로 사람 방향의 라이다 거리를 계산합니다.
- 화면 창이 느리거나 필요 없으면 `show_window = False`로 바꿉니다.
- 비상정지는 실습용 소프트웨어 기능입니다. 사람이 있는 곳에서는 로봇 전원을 바로 끌 수 있게 준비하세요.
