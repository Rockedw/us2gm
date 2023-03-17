#
# Copyright:: Copyright (c) Chef Software Inc.
# License:: Apache License, Version 2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

require "spec_helper"

describe Chef::Resource::ChefVaultSecret do
  let(:resource) { Chef::Resource::ChefVaultSecret.new("foo") }

  it "has a resource name of :chef_vault_secret" do
    expect(resource.resource_name).to eql(:chef_vault_secret)
  end

  it "sets the default action as :create" do
    expect(resource.action).to eql([:create])
  end

  it "id is the name property" do
    expect(resource.id).to eql("foo")
  end

  it "supports :create, :create_if_missing, and :delete actions" do
    expect { resource.action :create }.not_to raise_error
    expect { resource.action :create_if_missing }.not_to raise_error
    expect { resource.action :delete }.not_to raise_error
  end
end
