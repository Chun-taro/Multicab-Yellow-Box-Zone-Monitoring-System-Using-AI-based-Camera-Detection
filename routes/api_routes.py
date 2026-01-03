from flask import Blueprint, jsonify
from database.database import Database

api_bp = Blueprint('api', __name__)

@api_bp.route('/violations')
def get_violations():
    db = Database()
    violations = db.get_all_violations()
    db.close()
    return jsonify(violations)
