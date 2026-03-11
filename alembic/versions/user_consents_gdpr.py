"""
Revision ID: user_consents_gdpr
Revises: 
Create Date: 2026-03-11
"""
from alembic import op
import sqlalchemy as sa

revision = 'user_consents_gdpr'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "user_consents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, nullable=False), 
        sa.Column("consent_type", sa.String, nullable=False),
        sa.Column("granted_at", sa.DateTime, nullable=True),
        sa.Column("revoked_at", sa.DateTime, nullable=True),
    )

def downgrade():
    op.drop_table('user_consents')
