from flask import Blueprint
from controllers.auth import Login, Register
from controllers.conversation import SendConversation

public_bp = Blueprint('public', __name__,url_prefix='/public')

# Register the Login and Register classes for public routes
public_bp.add_url_rule('/login', view_func=Login.as_view('login'))
public_bp.add_url_rule('/register', view_func=Register.as_view('register'))
