class AddEnabledBookmarkTypesToAdminProjects < ActiveRecord::Migration[5.1]
  def up
    add_column :admin_projects, :enabled_bookmark_types, :text
  end

  def down
    remove_column :admin_project, :enabled_bookmark_types
  end
end
