"""
Revision ID: add_audit_log
Revises: user_consents_gdpr
Create Date: 2026-03-11
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_audit_log'
down_revision = 'user_consents_gdpr'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, nullable=False, index=True),
        sa.Column('action', sa.String, nullable=False),
        sa.Column('detail', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )


def downgrade():
    op.drop_table('audit_logs')
