# Настройка
* Настройка производится в cmd. 1) открыть cmd 2) зайти в папку с проектом
* Если настройка производится в powershell(от имени администратора) прописать доступ **Set-ExecutionPolicy RemoteSigned** и поставить значение A - для всех
* Установить пакет вирутуальной среды pip install virtualenv
* Создать виртуальное окружение python -m venv env
* Активировать env\Scripts\activate.bat (для cmd) / env\Scripts\activate.ps1 (для powershell)
* Установить зависимости pip install -r requirements.txt


# Train model 
* **python main.py -m train**

# Test model  "testfile" 
Input data for test is in file **testfile**
* **python main.py -m test**
