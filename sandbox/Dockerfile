# run_tests 的沙箱:只多裝 pytest。正式環境請改成預裝該 repo 相依套件的映像。
FROM python:3.13-slim
RUN pip install --no-cache-dir pytest==8.*
RUN useradd --create-home sandbox
USER sandbox
