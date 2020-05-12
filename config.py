import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'gsolvit'

    @staticmethod
    def init_app(app):
        pass

config = {
    'default': Config,
    'MYSQL_HOST': 'localhost',
    'MYSQL_PORT': 8889,
    'MYSQL_USERNAME': 'scm',
    'MYSQL_PASSWORD': 'scm',
    'DATABASE_NAME': 'studentCourseManagement'
}