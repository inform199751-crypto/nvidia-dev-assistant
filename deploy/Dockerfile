# 小助理多人部署的映像:跑 webui.py --team,前面一定要接公司的登入代理(見 deploy/README.md)。
# 建置(在專案根目錄):docker build -f deploy/Dockerfile -t dev-assistant .
FROM docker:27.5.1-cli AS dockercli

FROM python:3.13-slim
# git:propose_patch 用 git apply 驗證 patch、git_log / git_blame;
# docker CLI:run_tests / run_command 透過主機的 Docker 開沙箱(要另外掛 docker.sock,預設不開)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
# 工作區是從主機唯讀掛進來的,擁有者跟容器裡的使用者不同,git 會拒絕(dubious ownership);上面那行放行
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py models.yaml webui.html ./
# 子資料夾要另外列:ml/ 是 agent.py 一啟動就 import 的(少了它整個服務起不來),assets/ 是頁面的圖示
COPY ml/*.py ml/
COPY assets/ assets/

RUN useradd --create-home app && mkdir -p /data/output && chown app /data/output
USER app
ENV TZ=Asia/Taipei PYTHONUTF8=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 \
    AGENT_WORKSPACE=/workspace AGENT_OUTPUT=/data/output
EXPOSE 8765
CMD ["python", "webui.py", "--team", "--port", "8765"]
