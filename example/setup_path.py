import os
import sys

# 현재 파일의 디렉토리 경로를 기준으로 'gradient'와 'functions' 디렉토리를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'gradient'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))