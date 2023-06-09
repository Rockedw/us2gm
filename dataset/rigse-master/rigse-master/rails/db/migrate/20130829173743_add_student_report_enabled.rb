class AddStudentReportEnabled < ActiveRecord::Migration[5.1]
  def change
    add_column :investigations,      :student_report_enabled, :boolean, :default => true
    add_column :activities,          :student_report_enabled, :boolean, :default => true
    add_column :external_activities, :student_report_enabled, :boolean, :default => true
  end
end
