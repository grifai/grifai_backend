<%text>#
# Alembic migration script
#
</%text>

revision = '${up_revision}'
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

from alembic import op
import sqlalchemy as sa

${upgrades if upgrades else "# No upgrade steps"}

def downgrade():
${downgrades if downgrades else "    # No downgrade steps"}
