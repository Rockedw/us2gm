class RenameNoticeUserDisplayStatus < ActiveRecord::Migration[5.1]
  def up
    rename_table :notice_user_display_statuses, :admin_notice_user_display_statuses
  end

  def down
    rename_table :admin_notice_user_display_statuses, :notice_user_display_statuses
  end
end
