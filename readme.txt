■Githubコマンド
git init
git add .
git commit -m "initial fastapi commit"

git remote add origin https://github.com/lzqdiy/mypython.git

git branch
git branch -M main


git push -u origin main


■Local　Test
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 5000


■render設定　https://dashboard.render.com/web/new?onboarding=active
<python3.13.4>
pip install -r requirements.txt
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 5000

■
ctrl +shift +P 
create Python虚拟环境   

