class AddAutoPortalAuthorSetting < ActiveRecord::Migration[5.1]
  def up
    add_column :admin_settings, :auto_set_teachers_as_authors, :boolean, :default => false
  end

  def down
    remove_column :admin_settings, :auto_set_teachers_as_authors
  end
end
