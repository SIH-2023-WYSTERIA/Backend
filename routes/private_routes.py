from flask import Blueprint
from controllers.conversation import (
    SendConversation,
    GetAllConversations,
    GetConversationsByEmail,
    SendFinetuneData,
    GetFinetuneData,
    FinetuneData,
)
from controllers.employee import GetEmployeeByEmail
from controllers.company import GetStats

private_bp = Blueprint("private", __name__, url_prefix="/private")

private_bp.add_url_rule(
    "/send_conversation", view_func=SendConversation.as_view("send_conversation")
)
private_bp.add_url_rule(
    "/get_all_conversations",
    view_func=GetAllConversations.as_view("get_all_conversations"),
)
private_bp.add_url_rule(
    "/get_emp_by_email", view_func=GetEmployeeByEmail.as_view("get_emp_by_email")
)
private_bp.add_url_rule(
    "/get_conv_by_email", view_func=GetConversationsByEmail.as_view("get_conv_by_email")
)
private_bp.add_url_rule("/get_stats", view_func=GetStats.as_view("get_stats"))
private_bp.add_url_rule(
    "/send_finetune_data", view_func=SendFinetuneData.as_view("send_finetune_data")
)
private_bp.add_url_rule(
    "/get_finetune_data", view_func=GetFinetuneData.as_view("get_finetune_data")
)
private_bp.add_url_rule(
    "/finetune_data", view_func=FinetuneData.as_view("finetune_data")
)
