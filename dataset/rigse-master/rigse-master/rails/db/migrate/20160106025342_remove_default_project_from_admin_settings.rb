class RemoveDefaultProjectFromAdminSettings < ActiveRecord::Migration[5.1]
  def up
    remove_column :admin_settings, :default_project_id
  end

  def down
    add_column :admin_settings, :default_project_id, :integer
  end
end
