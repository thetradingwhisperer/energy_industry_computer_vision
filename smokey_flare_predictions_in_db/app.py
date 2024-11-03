from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.String(80), nullable=False)
    bbox_x = db.Column(db.Float, nullable=False)
    bbox_y = db.Column(db.Float, nullable=False)
    bbox_width = db.Column(db.Float, nullable=False)
    bbox_height = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/predictions', methods=['POST'])
def add_prediction():
    data = request.get_json()
    new_prediction = Prediction(
        class_name=data['class'],
        bbox_x=data['bbox']['x'],
        bbox_y=data['bbox']['y'],
        bbox_width=data['bbox']['width'],
        bbox_height=data['bbox']['height']
    )
    db.session.add(new_prediction)
    db.session.commit()
    return jsonify({'message': 'Prediction saved'}), 201

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
