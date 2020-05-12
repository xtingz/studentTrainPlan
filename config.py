import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'gsolvit'

    @staticmethod
    def init_app(app):
        pass

config = {
    'default': Config,
    'MYSQL_PATH': 'localhost',
    'MYSQL_USERNAME': 'test',
    'MYSQL_PASSWORD': 'test',
    'DATABASE_NAME': 'studentTrainPlan'
}
