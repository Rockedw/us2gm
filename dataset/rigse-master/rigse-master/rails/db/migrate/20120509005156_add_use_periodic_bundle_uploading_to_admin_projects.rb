class AddUsePeriodicBundleUploadingToAdminProjects < ActiveRecord::Migration[5.1]
  def self.up
    add_column :admin_projects, :use_periodic_bundle_uploading, :boolean, :default => false
  end

  def self.down
    remove_column :admin_projects, :use_periodic_bundle_uploading
  end
end
