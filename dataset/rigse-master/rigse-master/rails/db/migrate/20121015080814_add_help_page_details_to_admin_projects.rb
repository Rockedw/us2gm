class AddHelpPageDetailsToAdminProjects < ActiveRecord::Migration[5.1]
  def change
    add_column :admin_projects, :external_url, :string
    add_column :admin_projects, :custom_help_page_html, :text
    add_column :admin_projects, :help_type, :boolean
  end
end
