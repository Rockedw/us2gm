class AddHomePageContentToAdminProjects < ActiveRecord::Migration[5.1]
  def self.up
    add_column :admin_projects, :home_page_content, :text
  end

  def self.down
    drop_column :admin_projects, :home_page_content
  end
end
