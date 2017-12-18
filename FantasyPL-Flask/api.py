# Imports
from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse
from flaskext.mysql import MySQL

# Initiate Flask application and MySQL database variable
app = Flask(__name__)
api = Api(app)
mysql = MySQL()

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'alexB'
app.config['MYSQL_DATABASE_PASSWORD'] = 'beans'
app.config['MYSQL_DATABASE_DB'] = 'ItemListDb'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

# Class for user creation
class CreateUser(Resource):
    def post(self):
        try:
            # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('email', type=str, help='Email address to create user')
            parser.add_argument('password', type=str, help='Password to create user')
            args = parser.parse_args()
            # User's details as the args
            _userEmail = args['email']
            _userPassword = args['password']
            # Connection to the database and use a cursor to execute the call
            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.callproc('spCreateUser', (_userEmail, _userPassword))
            data = cursor.fetchall()
            # Send back http response if successful or not
            if len(data) is 0:
                conn.commit()
                return {'StatusCode': '200', 'Message': 'User creation success'}
            else:
                return {'StatusCode': '1000', 'Message': str(data[0])}
        except Exception as e:
            return {'error': str(e)}


class AuthenticateUser(Resource):
    def post(self):
        try:
            # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('email', type=str, help='Email address for Authentication')
            parser.add_argument('password', type=str, help='Password for Authentication')
            args = parser.parse_args()

            _userEmail = args['email']
            _userPassword = args['password']

            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.callproc('sp_AuthenticateUser', (_userEmail,))
            data = cursor.fetchall()

            if (len(data) > 0):
                if (str(data[0][2]) == _userPassword):
                    return {'status': 200, 'UserId': str(data[0][0])}
                else:
                    return {'status': 100, 'message': 'Authentication failure'}

        except Exception as e:
            return {'error': str(e)}


class getPrototype(Resource):
    def get(self):
        # Parse the arguments
        parser = reqparse.RequestParser()
        parser.add_argument('predictions', type=str, help='Button to get predictions')
        args = parser.parse_args()

        _predictions = args['predictions']

        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.callproc('sp_getPredictions', (_predictions))
        data = cursor.fetchall()



class AddItem(Resource):
    def post(self):
        try:
            # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('id', type=str)
            parser.add_argument('item', type=str)
            args = parser.parse_args()

            _userId = args['id']
            _item = args['item']

            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.callproc('sp_AddItems',(_userId,_item))
            data = cursor.fetchall()

            conn.commit()
            return {'StatusCode':'200','Message': 'Success'}

        except Exception as e:
            return {'error': str(e)}

class GetAllItems(Resource):
    def post(self):
        try:
            # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('id', type=str)
            args = parser.parse_args()

            _userId = args['id']

            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.callproc('sp_GetAllItems',(_userId,))
            data = cursor.fetchall()

            items_list=[]
            for item in data:
                i = {
                    'Id':item[0],
                    'Item':item[1]
                }
                items_list.append(i)

            return {'StatusCode':'200','Items':items_list}

        except Exception as e:
            return {'error': str(e)}

api.add_resource(CreateUser, '/CreateUser')
api.add_resource(AuthenticateUser, '/AuthenticateUser')
api.add_resource(GetAllItems, '/GetAllItems')

# if __name__ == '__main__':
#     app.run(debug=True)
