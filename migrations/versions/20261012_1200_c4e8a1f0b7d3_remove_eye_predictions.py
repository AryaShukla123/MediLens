"""remove retinal-imaging (eye) predictions

Revision ID: c4e8a1f0b7d3
Revises: 1a167640d4cf
Create Date: 2026-10-12 12:00:00.000000

The retinal-imaging module was removed from MediLens. This deletes any eye
screenings saved before that (and the chat messages attached to them), so no
orphaned health data stays behind in the database.

Nothing else changes: no tables or columns are touched. A fresh database has no
such rows, so this does nothing there.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4e8a1f0b7d3'
down_revision: Union[str, None] = '1a167640d4cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text(
        "DELETE FROM chat_messages WHERE prediction_id IN "
        "(SELECT id FROM predictions WHERE module = 'eye')"
    ))
    op.execute(sa.text("DELETE FROM predictions WHERE module = 'eye'"))


def downgrade() -> None:
    # Deleted rows cannot be restored.
    pass
