class AddIsLeafToStandardStatements < ActiveRecord::Migration[5.1]
  def change
    add_column :standard_statements, :is_leaf, :boolean
  end
end
