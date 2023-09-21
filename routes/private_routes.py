from flask import Blueprint
from controllers.conversation import SendConversation, GetAllConversations, GetConversationsByEmail
from controllers.employee import GetEmployeeByEmail

private_bp = Blueprint('private', __name__,url_prefix='/private')

private_bp.add_url_rule('/send_conversation', view_func=SendConversation.as_view('send_conversation'))
private_bp.add_url_rule('/get_all_conversations', view_func=GetAllConversations.as_view('get_all_conversations'))
private_bp.add_url_rule('/get_emp_by_email', view_func=GetEmployeeByEmail.as_view('get_emp_by_email'))
private_bp.add_url_rule('/get_conv_by_email', view_func=GetConversationsByEmail.as_view('get_conv_by_email'))