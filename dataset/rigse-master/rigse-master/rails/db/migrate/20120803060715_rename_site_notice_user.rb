class RenameSiteNoticeUser < ActiveRecord::Migration[5.1]
  def up
    rename_table :site_notice_users, :admin_site_notice_users
  end

  def down
    rename_table :admin_site_notice_users, :site_notice_users
  end
end
