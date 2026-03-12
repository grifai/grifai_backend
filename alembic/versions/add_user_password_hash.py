"""Add password_hash column to users table

Revision ID: add_user_password_hash
Revises: add_audit_log
Create Date: 2026-03-12
"""
import sqlalchemy as sa
from alembic import op

revision = "add_user_password_hash"
down_revision = "add_audit_log"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("password_hash", sa.String(), nullable=True))


def downgrade():
    op.drop_column("users", "password_hash")
