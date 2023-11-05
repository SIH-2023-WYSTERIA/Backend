from flask import Blueprint
from controllers.auth import (
    CompanyLogin,
    EmployeeLogin,
    RegisterCompany,
    RegisterEmployee,
)
from controllers.employee import GetEmployee
from controllers.chat_proxy import Chat, ChatFinetune

public_bp = Blueprint("public", __name__, url_prefix="/public")

# Register the Login and Register classes for public routes
public_bp.add_url_rule("/company/login", view_func=CompanyLogin.as_view("company_login"))
public_bp.add_url_rule("/employee/login", view_func=EmployeeLogin.as_view("employee_login"))
public_bp.add_url_rule("/company/register", view_func=RegisterCompany.as_view("register_company"))
public_bp.add_url_rule("/employee/register", view_func=RegisterEmployee.as_view("register_employee"))
public_bp.add_url_rule("/employee/get_employee", view_func=GetEmployee.as_view("get_employee"))
public_bp.add_url_rule("/chat", view_func=Chat.as_view("chat"))
public_bp.add_url_rule("/chat_finetune", view_func=ChatFinetune.as_view("chat_finetune"))
